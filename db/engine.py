from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path
from sqlalchemy import event

import models.schema  # noqa: F401


def get_database_path() -> str:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return str(data_dir / "standups.db")


def create_db_engine():
    db_path = get_database_path()
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False}
    )

    # Enable foreign key constraints
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine

engine = create_db_engine()


def create_all_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session