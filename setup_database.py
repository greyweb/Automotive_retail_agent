import sqlite3
import pandas as pd

csv_file = "car_inventory_filled.csv"
db_file = "car_inventory.db" 
table_name = "car_details"       

# Read CSV into DataFrame
df = pd.read_csv(csv_file)

# Connect to SQLite DB (creates file if it doesn't exist)
conn = sqlite3.connect(db_file)

# Write DataFrame to SQLite (replace if table already exists)
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Commit & close
conn.commit()
conn.close()

print(f"CSV '{csv_file}' uploaded into table '{table_name}' in '{db_file}'")