import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Load environment variables
load_dotenv()

raw_password = os.getenv("DATABASE_PW")
if not raw_password:
  raise ValueError("DATABASE_PW environment variable is missing!")

encoded_password = urllib.parse.quote_plus(raw_password)

# Supabase PostgreSQL Connection String
DATABASE_URL = f"postgresql://postgres.avmyohbogdlamvppmesm:{encoded_password}@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres"

# Initialize SQLAlchemy Engine
engine = create_engine(DATABASE_URL, poolclass=NullPool, echo=False)

# Session factory for creating fresh, isolated sessions
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
  """Generates a fresh database session and ensures it closes properly."""
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()


# Connection test block (only runs if connection.py is executed directly)
if __name__ == "__main__":
  try:
    with engine.connect() as connection:
      result = connection.execute(text("SELECT version();"))
      print("Connected successfully!")
      print("PostgreSQL Version:", result.fetchone()[0])
  except Exception as e:
    print(f"An error occurred: {e}")