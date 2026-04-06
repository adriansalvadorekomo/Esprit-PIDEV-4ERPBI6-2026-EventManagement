import pandas as pd
from sqlalchemy import create_engine, text

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
df = pd.read_sql(text(SQL_QUERY), engine)

print("--- Data Summary ---")
print(df.describe())
print("\n--- Missing Dates Check ---")
df['ds'] = pd.to_datetime(df['ds'])
all_dates = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
missing = all_dates.difference(df['ds'])
print(f"Total days in range: {len(all_dates)}")
print(f"Days with data: {len(df)}")
print(f"Missing days: {len(missing)}")

print("\n--- Zero Values Check ---")
print(f"Days with 0 reservations: {(df['y'] == 0).sum()}")

print("\n--- Head ---")
print(df.head(10))
