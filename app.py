from datetime import datetime
from sqlalchemy import create_engine, func, text, select, Integer, BigInteger, Double, Boolean, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import select, insert
import urllib.parse
import os
from dotenv import load_dotenv
import pandas as pd

