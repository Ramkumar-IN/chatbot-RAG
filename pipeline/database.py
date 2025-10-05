import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

print("DEBUG password from env:", os.getenv("PGVECTOR_PASSWORD_KEY"))  # 👈 add this
def connect_pg():
    conn = psycopg2.connect(
        host="postgres-vectorstore.clmnkejpqspy.us-east-1.rds.amazonaws.com",
        port=5432,
        dbname="postgres",
        user="postgres",
        password=os.getenv("PGVECTOR_PASSWORD_KEY")
    )
    return conn
