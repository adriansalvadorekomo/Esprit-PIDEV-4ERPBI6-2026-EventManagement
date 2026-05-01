import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd
import numpy as np
import pickle, os, datetime, warnings
import psycopg2, mlflow
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm

warnings.filterwarnings("ignore")

# ─── MLflow ─────────────────────────────────────
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ts_final_price_forecast")
mlflow.autolog(disable=True)
mlflow.statsmodels.autolog(disable=True)

# ─── DB ─────────────────────────────────────────
from settings import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
query = """
SELECT e.event_date, f.final_price
FROM fact_suivi_event f
LEFT JOIN dim_event e ON f.event_sk = e.event_sk
WHERE e.event_date IS NOT NULL
"""
df = pd.read_sql(query, conn)
conn.close()

df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
df = df.dropna(subset=["event_date"])

ts_raw = (
    df.set_index("event_date")["final_price"]
    .resample("MS").sum().asfreq("MS")
)
# Replace zeros/NaN with interpolated values to avoid flat-line model
ts = ts_raw.replace(0, np.nan).interpolate(method="time").ffill().bfill()
print(f"Series: {len(ts)} months  ({ts.index[0].date()} → {ts.index[-1].date()})")

# ════════════════════════════════════════════════
# 1. STATIONARITY — ADF test (+ auto-differencing)
# ════════════════════════════════════════════════
def adf_report(series, label="series"):
    result = adfuller(series.dropna())
    print(f"\nADF [{label}]  stat={result[0]:.4f}  p={result[1]:.4f}  "
          f"{'✅ stationary' if result[1] <= 0.05 else '❌ non-stationary'}")
    return result[1]

p_val = adf_report(ts, "raw")
diff_order = 0
ts_check = ts.copy()
while p_val > 0.05 and diff_order < 3:
    diff_order += 1
    ts_check = ts_check.diff().dropna()
    p_val = adf_report(ts_check, f"diff={diff_order}")

print(f"→ Suggested d = {diff_order}")

# ════════════════════════════════════════════════
# 2. TRANSFORMATION — Box-Cox (requires all > 0)
# ════════════════════════════════════════════════
offset = 1.0  # avoids log(0)
ts_pos = ts + offset

ts_transformed, lam = stats.boxcox(ts_pos)
ts_transformed = pd.Series(ts_transformed, index=ts.index)
print(f"\nBox-Cox λ = {lam:.4f}  (λ≈0 ≈ log, λ≈1 ≈ no transform)")

# ─── Train / test split ─────────────────────────
n_test = 3
train_t = ts_transformed.iloc[:-n_test]
test_t  = ts_transformed.iloc[-n_test:]
test_orig = ts.iloc[-n_test:]

# ════════════════════════════════════════════════
# 3. AUTO_ARIMA — optimal (p,d,q)(P,D,Q,m)
# ════════════════════════════════════════════════
print("\nRunning auto_arima …")
n_train = len(train_t)
# Seasonal ARIMA requires >> m observations; skip seasonality for short series
use_seasonal = n_train >= 3 * 12

auto = pm.auto_arima(
    train_t,
    m=12 if use_seasonal else 1,
    seasonal=use_seasonal,
    stepwise=True,
    information_criterion="aicc",
    max_p=4, max_q=4,
    start_P=0, max_P=2 if use_seasonal else 0,
    start_Q=0, max_Q=2 if use_seasonal else 0,
    d=None, D=1 if use_seasonal else 0,
    test="adf",
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    n_jobs=-1,
)
print(f"\nBest order: {auto.order}  seasonal: {auto.seasonal_order}")
order          = auto.order
seasonal_order = auto.seasonal_order

# ════════════════════════════════════════════════
# 4. FIT + RESIDUAL DIAGNOSTICS
# ════════════════════════════════════════════════
model  = SARIMAX(train_t, order=order, seasonal_order=seasonal_order,
                 enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)
print(result.summary())

os.makedirs("models", exist_ok=True)
diag_path = "models/sarima_diagnostics.png"
try:
    fig = result.plot_diagnostics(figsize=(12, 8))
    fig.savefig(diag_path, bbox_inches="tight")
    plt.close()
    print(f"\nDiagnostics saved → {diag_path}")
except ValueError as e:
    print(f"\nDiagnostics skipped (series too short): {e}")

# ════════════════════════════════════════════════
# 5. EVALUATE — inverse Box-Cox → original scale
# ════════════════════════════════════════════════
def inv_boxcox(y, lam, offset=1.0):
    """Reverse Box-Cox and remove the offset."""
    if abs(lam) < 1e-10:
        return np.exp(y) - offset
    base = lam * np.asarray(y) + 1
    base = np.maximum(base, 0)  # clamp to avoid NaN from negative base
    return np.power(base, 1.0 / lam) - offset

forecast_t   = result.forecast(steps=n_test)
forecast_orig = inv_boxcox(forecast_t.values, lam, offset)
forecast_orig = np.maximum(forecast_orig, 0)   # clip negatives

mae  = mean_absolute_error(test_orig, forecast_orig)
rmse = np.sqrt(mean_squared_error(test_orig, forecast_orig))
mape = np.mean(np.abs((test_orig.values - forecast_orig) /
                       np.maximum(np.abs(test_orig.values), 1))) * 100

print(f"\n📊 Test metrics (original scale):")
print(f"   MAE  = {mae:,.2f}")
print(f"   RMSE = {rmse:,.2f}")
print(f"   MAPE = {mape:.2f}%")
print(f"   AIC  = {result.aic:.2f}")

# ─── Refit on full transformed series ───────────
final_model  = SARIMAX(ts_transformed, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
final_result = final_model.fit(disp=False)

# ─── Future forecast (6 months, original scale) ─
horizon = 6
future_t = final_result.forecast(steps=horizon)
future_orig = inv_boxcox(future_t.values, lam, offset)
future_orig = np.nan_to_num(future_orig, nan=0.0, posinf=0.0, neginf=0.0)
future_orig = np.maximum(future_orig, 0)
future_index = pd.date_range(ts.index[-1] + pd.DateOffset(months=1),
                              periods=horizon, freq="MS")
forecast_df = pd.DataFrame({"ds": future_index, "yhat": future_orig})

forecast_path = "models/ts_forecast.csv"
forecast_df.to_csv(forecast_path, index=False)
print("\nForecast next 6 months:")
print(forecast_df.to_string(index=False))

# ─── MLflow run ─────────────────────────────────
version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
mlflow.end_run()
with mlflow.start_run():
    mlflow.log_params({
        "arima_p": order[0], "arima_d": order[1], "arima_q": order[2],
        "sarima_P": seasonal_order[0], "sarima_D": seasonal_order[1],
        "sarima_Q": seasonal_order[2], "sarima_m": seasonal_order[3],
        "boxcox_lambda": round(lam, 4), "n_test_months": n_test,
    })
    mlflow.log_metrics({
        "mae": round(mae, 2), "rmse": round(rmse, 2),
        "mape": round(mape, 2), "aic": round(result.aic, 2),
    })
    if os.path.exists(diag_path):
        mlflow.log_artifact(diag_path, artifact_path="diagnostics")

# ─── Persist model + lambda ─────────────────────
payload = {"model": final_result, "boxcox_lambda": lam, "offset": offset}
with open(f"models/sarima_model_{version}.pkl", "wb") as f:
    pickle.dump(payload, f)
with open("sarima_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"\n✅ TS training done — version: {version}")
