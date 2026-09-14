from sqlalchemy import create_engine, text
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()
raw_password = os.getenv("DATABASE_PW")
encoded_password = urllib.parse.quote_plus(raw_password)

# Get connection string from supabase
# Format: postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = f"postgresql+psycopg2://myuser:{encoded_password}@localhost:5432/mydatabase"
# initialise the sqlalchemy engine
engine = create_engine(DATABASE_URL)

# Test the connection + error handling
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print("Connected successfully!")
        print("PostgreSQL Version:", result.fetchone()[0])
except Exception as e:
    print(f"An error occurred: {e}")