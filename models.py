import asyncio

from sqlalchemy import (
    Column, Integer, String, PrimaryKeyConstraint, text,
    DateTime, Text, ForeignKeyConstraint
)
from sqlalchemy.dialects.postgresql import JSONB, BYTEA
from sqlalchemy.orm import declarative_base

import config
from database import get_db_engine

ModelBase = declarative_base()


class Documents(ModelBase):
    __tablename__ = 'documents'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='document_pkey'),
    )

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP(0)'))
    updated_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP(0)'))


class Models(ModelBase):
    __tablename__ = 'models'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='model_pkey'),
    )

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    training_parameters = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP(0)'))
    updated_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP(0)'))


class DocumentVectors(ModelBase):
    __tablename__ = 'document_vectors'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='document_vector_pkey'),
        ForeignKeyConstraint(['document_id'], ['documents.id']),
        ForeignKeyConstraint(['model_id'], ['models.id']),
    )

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        nullable=False,
        unique=True,
    )
    model_id = Column(
        Integer,
        nullable=False,
        unique=True,
    )

    vector = Column(BYTEA, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP(0)'))
    updated_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP(0)'))


async def ainit_models():
    engine = get_db_engine(config.Config.SQLALCHEMY_DATABASE_URI)

    async with engine.begin() as conn:
        await conn.run_sync(ModelBase.metadata.create_all)


def init_models():
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(ainit_models())
