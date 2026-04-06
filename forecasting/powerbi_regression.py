import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ─── 1. CONNECTION CONFIGURATION ───────────────────────────────────────────
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 5432,
    'database': 'DW_event',
    'user'    : 'postgres',
    'password': '1400'
}

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ─── 2. DATA EXTRACTION ──────────────────────────────────────────────────────
SQL_QUERY = """
SELECT
    dr.status,
    fs.price,
    de.event_date,
    c.category_name AS category
FROM public.fact_suivi_event fs
LEFT JOIN public.dim_reservation dr ON fs.reservation_sk = dr.reservation_sk
LEFT JOIN public.dim_event de ON fs.event_sk = de.event_sk
JOIN public."DIM_category" c ON fs.category_id = c.category_id
WHERE de.event_date IS NOT NULL;
"""

try:
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as conn:
        df_raw = pd.read_sql(text(SQL_QUERY), conn)
except Exception as e:
    print(f"Database connection error: {e}")
    df_raw = pd.DataFrame()

# ─── 3. FEATURE ENGINEERING (Monthly Aggregation) ──────────────────────────
if not df_raw.empty:
    df_raw['cancelled'] = df_raw['status'].apply(lambda x: 1 if str(x).lower() == 'cancelled' else 0)
    df_raw['price'] = df_raw['price'].fillna(df_raw['price'].median())
    df_raw['event_date'] = pd.to_datetime(df_raw['event_date'])
    
    # Global monthly trend for the model
    df_raw['month_period'] = df_raw['event_date'].dt.to_period('M')
    
    monthly = df_raw.groupby('month_period').agg(
        total_reservations=('status', 'count'),
        cancelled_count=('cancelled', 'sum'),
        avg_price=('price', 'mean'),
        std_price=('price', 'std'),
        max_price=('price', 'max'),
        min_price=('price', 'min')
    ).reset_index()
    
    # Fill missing months in the range
    start_month = monthly['month_period'].min()
    end_month = monthly['month_period'].max()
    all_months = pd.period_range(start=start_month, end=end_month, freq='M')
    monthly = monthly.set_index('month_period').reindex(all_months).fillna(0).reset_index()
    monthly.rename(columns={'index': 'month_period'}, inplace=True)

    monthly['cancellation_rate'] = np.where(monthly['total_reservations'] > 0, 
                                          monthly['cancelled_count'] / monthly['total_reservations'], 
                                          0)
    monthly['price_range'] = monthly['max_price'] - monthly['min_price']
    monthly['month_num'] = monthly['month_period'].dt.month
    monthly['quarter'] = monthly['month_period'].dt.quarter
    monthly['is_q4'] = (monthly['quarter'] == 4).astype(int)
    
    # Lag features
    monthly['cancel_rate_lag1'] = monthly['cancellation_rate'].shift(1).fillna(0)
    monthly['cancel_rate_lag2'] = monthly['cancellation_rate'].shift(2).fillna(0)
    
    # ─── 4. MODEL TRAINING ──────────────────────────────────────────────────
    FEATURE_COLS = [
        'total_reservations', 'avg_price', 'std_price', 'price_range',
        'month_num', 'quarter', 'is_q4', 'cancel_rate_lag1', 'cancel_rate_lag2'
    ]
    
    X = monthly[FEATURE_COLS].fillna(0)
    y = monthly['cancellation_rate']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Metrics calculation (Internal evaluation)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Robust MAPE: Only on months where rate > 0
    non_zero_mask = y > 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask])) * 100
    else:
        mape = 0 if np.sum(y_pred) == 0 else 100
    
    # Robust Accuracy: 1 - sum(abs(y - y_pred)) / sum(y)
    sum_actuals = np.sum(y)
    if sum_actuals > 0:
        accuracy = max(0, (1 - np.sum(np.abs(y - y_pred)) / sum_actuals) * 100)
    else:
        accuracy = 100 if np.sum(y_pred) == 0 else 0
    
    # ─── 5. FUTURE PREDICTION (Next Month KPI) ──────────────────────────────
    last_row = monthly.iloc[-1]
    next_month_date = (monthly['month_period'].max() + 1).to_timestamp()
    
    next_features = pd.DataFrame([{
        'total_reservations': monthly['total_reservations'].mean(),
        'avg_price': monthly['avg_price'].mean(),
        'std_price': monthly['std_price'].mean(),
        'price_range': monthly['price_range'].mean(),
        'month_num': next_month_date.month,
        'quarter': (next_month_date.month - 1) // 3 + 1,
        'is_q4': 1 if next_month_date.month >= 10 else 0,
        'cancel_rate_lag1': last_row['cancellation_rate'],
        'cancel_rate_lag2': last_row['cancel_rate_lag1']
    }])
    
    predicted_next_rate = model.predict(next_features[FEATURE_COLS])[0]
    
    # ─── 6. POWER BI OUTPUTS ────────────────────────────────────────────────
    
    # TABLE 1: KPIs & INDICATORS
    smart_target = 0.10
    deviation = predicted_next_rate - smart_target
    
    df_reg_kpis = pd.DataFrame({
        'Visual_Type': ['KPI', 'KPI', 'Indicator', 'Indicator'],
        'Name': ['Predicted Cancellation Rate', 'Deviation vs SMART Target', 'Model Reliability (R2)', 'Model Accuracy'],
        'Value': [f"{round(predicted_next_rate*100, 2)}%", f"{round(deviation*100, 2)}%", f"{round(r2*100, 1)}%", f"{round(accuracy, 1)}%"],
        'Status': ['Next Month Est.', 'Goal: 10%', 'Confidence Level', 'Overall Robustness']
    })

    # TABLE 2: TIME SERIES (Actual by Category vs Predicted Global)
    ts_actuals = df_raw.groupby(['month_period', 'category']).agg(
        actual_rate=('cancelled', 'mean')
    ).reset_index()
    ts_actuals['Date'] = ts_actuals['month_period'].dt.to_timestamp()
    
    monthly['Predicted_Rate'] = y_pred
    df_preds = monthly[['month_period', 'Predicted_Rate']]
    
    df_reg_visual_ts = pd.merge(ts_actuals, df_preds, on='month_period', how='left')
    df_reg_visual_ts.rename(columns={'actual_rate': 'Actual_Rate', 'category': 'Category'}, inplace=True)
    df_reg_visual_ts['Date'] = df_reg_visual_ts['month_period'].dt.to_timestamp()
    df_reg_visual_ts.drop(columns=['month_period'], inplace=True)

    print("✅ Regression Tables 'df_reg_kpis' and 'df_reg_visual_ts' are ready")
else:
    df_reg_kpis = pd.DataFrame()
    df_reg_visual_ts = pd.DataFrame()
    print("⚠️ No data extracted")
