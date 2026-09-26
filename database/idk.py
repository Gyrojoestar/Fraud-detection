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


query = "SELECT * FROM transactions WHERE is_labeled = true"