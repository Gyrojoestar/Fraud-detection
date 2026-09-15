from datetime import datetime
from sqlalchemy import create_engine, func, text, select, Integer, BigInteger, Double, Boolean, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from database.model import RawTransaction, UserFeatureStore
from database.connection import SessionLocal
from sqlalchemy import select, insert
import urllib.parse
import os
from dotenv import load_dotenv
import pandas as pd

def seed_database(csv_path: str):
    df = pd.read_csv(csv_path)

    # user_id with dummy data
    unique_user_ids = pd.DataFrame({'user_id': df['user_id'].unique()})

    unique_user_ids["tx_count_1h"] = 0
    unique_user_ids["tx_count_24h"] = 0
    unique_user_ids["avg_amount_1h"] = 0.0
    unique_user_ids["avg_amount_24h"] = 0.0
    unique_user_ids["last_tx_time"] = datetime.now()

    session = SessionLocal()
    try:
        # 1. Insert parent features
        session.execute(
            insert(UserFeatureStore), unique_user_ids.to_dict(orient='records')
        )

        # 2. Insert child raw transactions
        if 'transaction_id' in df.columns:
            df = df.drop(columns=['transaction_id'])
        session.execute(insert(RawTransaction), df.to_dict(orient='records'))

        session.commit()
        print('Database seeding completed successfully!')
    except Exception as e:
        session.rollback()
        print(f'Error during database seeding: {e}')
    finally:
        session.close()
    
csv_path = 'creditCardSupabase.csv'
if __name__ == "__main__":
    seed_database(csv_path)