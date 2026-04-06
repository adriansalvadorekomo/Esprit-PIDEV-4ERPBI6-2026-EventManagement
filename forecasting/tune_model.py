import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import warnings

warnings.filterwarnings('ignore')

def run_tuning():
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
        SUM(fs.reservations)                              AS y,
        SUM(fs.marketing_spend)                           AS marketing_spend,
        SUM(fs.visitors)                                  AS visitors
    FROM public.fact_suivi_event fs
    WHERE fs.reservation_date_fk IS NOT NULL
    GROUP BY ds
    ORDER BY ds ASC;
    """

    engine = create_engine(CONNECTION_STRING)
    df_raw = pd.read_sql(text(SQL_QUERY), engine, parse_dates=['ds'])

    # Aggregation
    df_daily_raw = df_raw.groupby('ds').agg({
        'y': 'sum',
        'marketing_spend': 'sum',
        'visitors': 'sum'
    }).reset_index()

    all_dates = pd.date_range(start=df_daily_raw['ds'].min(), end=df_daily_raw['ds'].max(), freq='D')
    df_daily = pd.DataFrame({'ds': all_dates})
    df_daily = df_daily.merge(df_daily_raw, on='ds', how='left').fillna(0)

    # Feature Engineering
    df_daily['marketing_lag7']  = df_daily['marketing_spend'].shift(7).fillna(0)
    df_daily['visitors_roll7']  = df_daily['visitors'].rolling(7, min_periods=1).mean().fillna(0)
    df_daily['is_weekend']      = (df_daily['ds'].dt.dayofweek >= 5).astype(int)

    REGRESSORS = ['marketing_lag7', 'visitors_roll7', 'is_weekend']

    # ─── 4. HYPERPARAMETER TUNING ──────────────────────────────────────────────
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['multiplicative', 'additive']
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    mapes = []

    print(f"Starting tuning for {len(all_params)} combinations...")

    for i, params in enumerate(all_params):
        m = Prophet(**params)
        for reg in REGRESSORS:
            m.add_regressor(reg)
        m.fit(df_daily)
        
        try:
            df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='30 days', parallel=None)
            df_p = performance_metrics(df_cv, rolling_window=1)
            
            # Check for MAPE or fallback to RMSE
            if 'mape' in df_p.columns:
                score = df_p['mape'].mean()
            else:
                score = df_p['rmse'].mean()
            
            mapes.append(score)
            print(f"[{i+1}/{len(all_params)}] Score: {score:.4f} | {params}")
        except Exception as e:
            print(f"[{i+1}/{len(all_params)}] Error: {e}")
            mapes.append(1e9)

    tuning_results = pd.DataFrame(all_params)
    tuning_results['score'] = mapes
    best_params = all_params[np.argmin(mapes)]

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best Score: {min(mapes):.4f}")

if __name__ == '__main__':
    run_tuning()
