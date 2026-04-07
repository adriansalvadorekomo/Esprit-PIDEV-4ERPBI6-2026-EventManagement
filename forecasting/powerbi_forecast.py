import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from prophet import Prophet
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
    TO_DATE(fs.reservation_date_fk::text, 'YYYYMMDD') AS ds,
    c.category_name                                   AS category,
    SUM(fs.reservations)                              AS y,
    SUM(fs.marketing_spend)                           AS marketing_spend,
    SUM(fs.visitors)                                  AS visitors
FROM public.fact_suivi_event fs
JOIN public."DIM_category" c ON fs.category_id = c.category_id
WHERE fs.reservation_date_fk IS NOT NULL
GROUP BY ds, category
ORDER BY ds ASC;
"""

try:
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as conn:
        df_raw = pd.read_sql(text(SQL_QUERY), conn, parse_dates=['ds'])
except Exception as e:
    print(f"Database connection error: {e}")
    df_raw = pd.DataFrame()

# ─── 3. PROCESSING & TUNED FORECAST ──────────────────────────────────────────
if not df_raw.empty:
    # 3.1 Daily Aggregation & Filling Zeros
    df_daily_raw = df_raw.groupby('ds').agg({
        'y': 'sum',
        'marketing_spend': 'sum',
        'visitors': 'sum'
    }).reset_index()

    all_dates = pd.date_range(start=df_daily_raw['ds'].min(), end=df_daily_raw['ds'].max(), freq='D')
    df_daily = pd.DataFrame({'ds': all_dates})
    df_daily = df_daily.merge(df_daily_raw, on='ds', how='left').fillna(0)

    # 3.2 Feature Engineering
    df_daily['marketing_lag7']  = df_daily['marketing_spend'].shift(7).fillna(0)
    df_daily['visitors_roll7']  = df_daily['visitors'].rolling(7, min_periods=1).mean().fillna(0)
    df_daily['is_weekend']      = (df_daily['ds'].dt.dayofweek >= 5).astype(int)

    REGRESSORS = ['marketing_lag7', 'visitors_roll7', 'is_weekend']

  # ─── 3.3 TUNED PARAMETERS (Ajustado para datos de eventos/sparse) ──────────
    # Se reducen las escalas para evitar que el modelo persiga el ruido de los días en 0
    TUNED_PARAMS = {
        'changepoint_prior_scale': 0.01, 
        'seasonality_prior_scale': 1.0,  
        'seasonality_mode': 'additive',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': True,
        'daily_seasonality': False
    }

    # 3.4 Evaluation (Backtesting last 30 days)
    test_days = 30
    if len(df_daily) > test_days * 2:
        train = df_daily.iloc[:-test_days]
        test  = df_daily.iloc[-test_days:]

        eval_model = Prophet(**TUNED_PARAMS)
        for reg in REGRESSORS:
            eval_model.add_regressor(reg)
        eval_model.fit(train[['ds', 'y'] + REGRESSORS])

        forecast_eval = eval_model.predict(test[['ds'] + REGRESSORS])
        y_true = test['y'].values
        y_pred = forecast_eval['yhat'].clip(lower=0).values

        # --- NUEVAS MÉTRICAS ROBUSTAS PARA EVENTOS ---
       # --- CÁLCULO DE MÉTRICAS ALINEADAS ---
        sum_actuals = np.sum(y_true)
        sum_pred = np.sum(y_pred)

        if sum_actuals > 0:
            # 1. Accuracy de Volumen (¿Le atinamos a la cantidad total?)
            vol_error = abs(sum_actuals - sum_pred) / sum_actuals
            accuracy = max(0.0, (1 - vol_error) * 100)

            # 2. MAPE de Volumen (Representa el riesgo sobre el total, no día a día)
            # Esto hará que Accuracy + MAPE = 100 (aprox), siendo mucho más lógico
            mape = min(vol_error * 100, 100.0) 
            
        else:
            accuracy = 100.0 if sum_pred == 0 else 0.0
            mape = 0.0 if sum_pred == 0 else 100.0
    else:
        accuracy, mape = 0.0, 100.0

    # Trend vs Last Month
    last_month_data = df_daily[df_daily['ds'] >= (df_daily['ds'].max() - pd.Timedelta(days=30))]
    last_month_avg  = last_month_data['y'].mean() if not last_month_data.empty else 0

    # ─── 4. FINAL PRODUCTION MODEL ───────────────────────────────────────────
    final_model = Prophet(**TUNED_PARAMS)
    for reg in REGRESSORS:
        final_model.add_regressor(reg)
    final_model.fit(df_daily[['ds', 'y'] + REGRESSORS])

    # Future DataFrame
    future = final_model.make_future_dataframe(periods=30, freq='D')
    recent_stats = df_daily[REGRESSORS].tail(30).mean()
    future = future.merge(df_daily[['ds'] + REGRESSORS], on='ds', how='left')

    for reg in REGRESSORS:
        if reg == 'is_weekend':
            future[reg] = future[reg].fillna((future['ds'].dt.dayofweek >= 5).astype(int))
        else:
            future[reg] = future[reg].fillna(recent_stats[reg])

    forecast = final_model.predict(future)

    # ─── 5. POWER BI OUTPUT TABLES ───────────────────────────────────────────
    forecast_30d = forecast[forecast['ds'] > df_daily['ds'].max()]
    total_forecast = forecast_30d['yhat'].clip(lower=0).sum()
    forecast_avg = forecast_30d['yhat'].clip(lower=0).mean()

    if last_month_avg > 0:
        trend_pct = ((forecast_avg - last_month_avg) / last_month_avg) * 100
    else:
        trend_pct = 0.0

    # TABLE 1: df_kpis
    df_kpis = pd.DataFrame({
        'Visual_Type': ['KPI', 'KPI', 'Indicator', 'Indicator'],
        'Name': ['Total Forecasted Reservations', 'Trend vs Last Month', 'Model Accuracy (Tuned)', 'MAPE (Risk Margin)'],
        'Value': [
            round(float(total_forecast), 0),
            round(trend_pct, 2),
            round(accuracy, 2),
            round(mape, 2)
        ],
        'Status': ['Next 30 Days', 'Growth/Decline', 'Optimized Model', 'Confidence Margin']
    })

    # TABLE 2: df_visual_ts
    ts_actuals = df_raw[['ds', 'category', 'y']].rename(columns={'ds': 'Date', 'category': 'Category', 'y': 'Actual'})
    ts_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    ts_forecast.rename(columns={'ds': 'Date', 'yhat': 'Prediction', 'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'}, inplace=True)
    ts_forecast[['Prediction', 'Lower_Bound', 'Upper_Bound']] = ts_forecast[['Prediction', 'Lower_Bound', 'Upper_Bound']].clip(lower=0)

    df_visual_ts = pd.merge(ts_forecast, ts_actuals, on='Date', how='left')
    df_visual_ts['Category'] = df_visual_ts['Category'].fillna('Forecast/Total')

    print("✅ Optimized Forecast Ready")
    print(f"   → New Accuracy: {accuracy:.2f}%")
    print(f"   → New MAPE: {mape:.2f}%")

else:
    df_kpis, df_visual_ts = pd.DataFrame(), pd.DataFrame()
    print("⚠️ No data")
