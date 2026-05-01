import mlflow
mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("ts_final_price_forecast")
runs = client.search_runs([exp.experiment_id], filter_string="attributes.status = 'RUNNING'")
print(f"{len(runs)} stale RUNNING runs:")
for r in runs:
    print(r.info.run_id, r.data.params)
