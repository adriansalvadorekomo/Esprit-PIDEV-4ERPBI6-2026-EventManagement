import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# 1️⃣ Rutas desde Talend
input_xlsx = sys.argv[1] # "archivo.xlsx"
output_csv = sys.argv[2] # "archivo_procesado.csv"

# 2️⃣ Leer el Excel original
df = pd.read_excel(input_xlsx, sheet_name='EVENT')

# 3️⃣ Ingeniería de variables (Mes del evento)
df['event_month'] = pd.to_datetime(df['event_date']).dt.month

# 4️⃣ Preparar columnas para el modelo
features = ['budget', 'type', 'event_month']
df_subset = df[features].copy()

# 5️⃣ Codificar el texto a números
encoder = OrdinalEncoder()
df_subset['type'] = encoder.fit_transform(df_subset[['type']].astype(str))

# 6️⃣ Configurar MissForest
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42
)

# 7️⃣ Imputar valores
imputed_data = imputer.fit_transform(df_subset)

# 8️⃣ Reintegrar el budget al DataFrame
df['budget'] = imputed_data[:, 0]

# 9️⃣ Limpiar (Quitar columna auxiliar)
df = df.drop(columns=['event_month'])

# 🔟 Guardar como CSV para Talend (Separado por '|')
df.to_csv(output_csv, sep="|", index=False, encoding="utf-8")
print(f"✅ Excel convertido e imputado a CSV: {output_csv}")