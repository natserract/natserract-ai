import logging
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine, AsyncConnection
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import Config


def get_db_engine(database_url: str) -> AsyncEngine:
    """
    Creates a SQLAlchemy session for the specified database URL.

    :param database_url: The URL of the database to connect to.
    :return: A tuple containing an instance of SQLAlchemy Session and an Engine object.
    """
    try:
        logging.info('trying to connect to database')
        async_engine = create_async_engine(
            database_url,
            future=True,
            # echo=True
        )

        logging.info(f"Connected to database successfully")
        return async_engine
    except SQLAlchemyError as e:
        logging.error(f"Error connecting to database: {str(e)}")
        raise e


def async_session_generator():
    engine = get_db_engine(
        Config.SQLALCHEMY_DATABASE_URI
    )

    return async_sessionmaker(
        engine,
        class_=AsyncSession
    )


@asynccontextmanager
async def get_session() -> AsyncSession:
    try:
        async_session = async_session_generator()

        async with async_session() as session:
            yield session
    except:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_connection() -> AsyncConnection:
    engine = get_db_engine(
        Config.SQLALCHEMY_DATABASE_URI
    )

    try:
        async with engine.connect() as connect:
            yield connect
    finally:
        await connect.close()


async def connect() -> asyncpg.connection.Connection:
    conn = await asyncpg.connect(
        dsn=Config.ASYNCPG_CONNECTION_URI
    )

    return conn
