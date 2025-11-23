# migrate.py
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 🔧 KONFIG DATABASE
# ==============================
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "database": os.getenv("DB_NAME", "apia_abiila_airq_db"),
    "user": os.getenv("DB_USER", "apia_abiila_admin"),
    "password": os.getenv("DB_PASS", "Criticaleven10"),
}

# ==============================
# 📦 DEFINISI SKEMA TABEL
# ==============================
TABLES = {}

# 1️⃣ Tabel data kualitas udara utama
TABLES["air_quality_data"] = """
    CREATE TABLE IF NOT EXISTS air_quality_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        waktu DATETIME NOT NULL,
        pm10 FLOAT,
        pm25 FLOAT,
        so2 FLOAT,
        co FLOAT,
        o3 FLOAT,
        no2 FLOAT,
        hc FLOAT,
        kelembaban FLOAT,
        suhu FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 2️⃣ Tabel forecast (untuk semua polutan)
pollutants = ["pm10", "pm25", "so2", "co", "o3", "no2", "hc"]

for pol in pollutants:
    TABLES[f"forecast_{pol}_data"] = f"""
        CREATE TABLE IF NOT EXISTS forecast_{pol}_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ds DATE,
            yhat FLOAT,
            yhat_lower FLOAT,
            yhat_upper FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    TABLES[f"forecast_{pol}_with_parameters_data"] = f"""
        CREATE TABLE IF NOT EXISTS forecast_{pol}_with_parameters_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ds DATE,
            yhat FLOAT,
            yhat_lower FLOAT,
            yhat_upper FLOAT,
            changepoint_prior_scale FLOAT,
            seasonality_prior_scale FLOAT,
            holidays_prior_scale FLOAT,
            model_mape FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

# 3️⃣ Tabel system history (Dashboard Status)
TABLES["system_status"] = """
    CREATE TABLE IF NOT EXISTS system_status (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME NOT NULL,
        backend VARCHAR(20),
        cpu_usage FLOAT,
        ram_usage FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 4️⃣ 🆕 Tabel Activity Log (dipakai Dashboard & ModelPage)
TABLES["activity_log"] = """
    CREATE TABLE IF NOT EXISTS activity_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        event VARCHAR(255) NOT NULL,
        detail TEXT,
        timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ==============================
# ⚙️ FUNGSI MIGRASI
# ==============================
def migrate():
    print("🔧 Connecting to database...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ Connected.\n")

    for table_name, sql in TABLES.items():
        print(f"➡️ Creating: {table_name:<45}", end="")
        try:
            cursor.execute(sql)
            print("✅")
        except Exception as e:
            print(f"❌ {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print("\n🎉 Migration completed successfully!\n")


if __name__ == "__main__":
    migrate()
