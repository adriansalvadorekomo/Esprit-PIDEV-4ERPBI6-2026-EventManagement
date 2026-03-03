import pandas as pd
import sys
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 1️⃣ Rutas desde Talend
input_xlsx = sys.argv[1]
output_csv = sys.argv[2]

# 2️⃣ Leer el Excel
df = pd.read_excel(input_xlsx)

# 3️⃣ Limpieza de tipos (Asegurar números)
cols = ['starting_price', 'rating_stars', 'value_money', 'functionality']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4️⃣ Seleccionar subset
df_subset = df[cols].copy()

# 5️⃣ Escalado (StandardScaler)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_subset)

# 6️⃣ Imputar con KNN
imputer = KNNImputer(n_neighbors=5, weights="distance")
imputed_array = imputer.fit_transform(df_scaled)

# 7️⃣ Des-escalar (Volver a valores originales)
df_values = pd.DataFrame(scaler.inverse_transform(imputed_array), columns=cols)

# 8️⃣ Reemplazar precio en el DataFrame original
df['starting_price'] = df_values['starting_price']

# 9️⃣ Guardar como CSV para Talend
df.to_csv(output_csv, sep="|", index=False, encoding="utf-8")
print(f"✅ Imputación KNN guardada en CSV para Talend")