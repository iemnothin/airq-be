# Production
# db.py
import mysql.connector

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="apia_abiila_admin",
            password="Criticaleven10",
            database="apia_abiila_airq_db"
        )
        return conn
    except Exception as e:
        print("❌ Database connection failed:", e)
        return None


# Local
# import mysql.connector

# def get_db_connection():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="",
#         database="db_airq"
#     )