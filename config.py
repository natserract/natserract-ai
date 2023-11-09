import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    APP_ENV = os.environ.get('APP_ENV')
    DB_HOST = os.environ.get('DB_HOST')
    DB_USER = os.environ.get('DB_USER')
    DB_PASS = os.environ.get('DB_PASS')
    DB_PORT = os.environ.get('DB_PORT')
    DB_DATABASE = os.environ.get('DB_DATABASE')
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
    ASYNCPG_CONNECTION_URI = os.environ.get('ASYNCPG_CONNECTION_URI')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL')
    VECTOR_STORE: str = os.environ.get('VECTOR_STORE')

