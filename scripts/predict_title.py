import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1️⃣ Charger le fichier
df = pd.read_csv("C:/Users/larry/Desktop/Esprit-PABI-4ERPBI6-2526-EventZella/data/service_raw.csv", sep="|")

# 2️⃣ Garder seulement les colonnes utiles
df = df[["id_service", "description", "title"]]

# 3️⃣ Renommer id_service → service_bkey
df = df.rename(columns={"id_service": "service_bkey"})

# 4️⃣ Nettoyer les valeurs incorrectes
df["title"] = df["title"].replace("#VALEUR!", pd.NA)

# 5️⃣ Séparer données connues et manquantes
df_train = df[df["title"].notna()].copy()
df_missing = df[df["title"].isna()].copy()

print("Nombre de titles manquants :", len(df_missing))

# 6️⃣ Nettoyer descriptions
df_train["description"] = df_train["description"].fillna("").astype(str)
df_missing["description"] = df_missing["description"].fillna("").astype(str)

# 7️⃣ Vectorisation TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X_train = vectorizer.fit_transform(df_train["description"])
y_train = df_train["title"]

# 8️⃣ Entraîner le modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9️⃣ Prédiction
if len(df_missing) > 0:
    X_missing = vectorizer.transform(df_missing["description"])
    predicted_titles = model.predict(X_missing)

    # Remplacer les valeurs manquantes
    df.loc[df["title"].isna(), "title"] = predicted_titles
    print("Imputation terminée ")
else:
    print("Aucun title manquant ")

# 🔟 Afficher le résultat final
df_final = df[["service_bkey", "description", "title"]]
print(df_final.head())

# 1️⃣1️⃣ Sauvegarder le fichier
df_final.to_csv(
    "C:/Users/larry/Desktop/Esprit-PABI-4ERPBI6-2526-EventZella/data/service_imputed.csv",
    index=False,
    encoding="utf-8",
    sep="|"
)

print("Fichier sauvegardé avec succès ")