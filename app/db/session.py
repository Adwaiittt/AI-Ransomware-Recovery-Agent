"""Database engine/session management."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


def make_engine(database_url: str) -> Engine:
    """Create an engine; for SQLite, ensure the parent dir exists and enable FKs."""
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        # FastAPI runs sync endpoints in a threadpool, so the connection may be
        # used from a different thread than the one that created it.
        connect_args["check_same_thread"] = False
        db_path = database_url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url, connect_args=connect_args, future=True)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            # WAL lets the API read while the watcher (Phase 2) writes.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create tables if they do not exist (no migrations tool needed at this scale)."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
