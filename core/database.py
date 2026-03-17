import os
import psycopg2
from yoyo import read_migrations, get_backend
from psycopg2 import pool
from psycopg2.pool import ThreadedConnectionPool
import core.config as config

class Database:

    def __init__(self) -> None:
        self.host = config.Config.database.host
        self.port = config.Config.database.port
        self.user = config.Config.database.user
        self.password = config.Config.database.password
        self.database = config.Config.database.database
        self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self._pool: ThreadedConnectionPool | None = None
    
    def connect(self):
        self._pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def get_conn(self) -> psycopg2.extensions.connection:
        if not self._pool:
            raise RuntimeError("Database pool not initialized")
        return self._pool.getconn()

    def put_conn(self, conn):
        if self._pool:
            self._pool.putconn(conn)

    def close(self):
        if self._pool:
            self._pool.closeall()

    def init_database(self) -> ThreadedConnectionPool:
        self.connect()
        self._run_migrations()
        return self._pool

    def _run_migrations(self) -> None:
        migrations_path = os.path.join(os.path.dirname(__file__), "../migrations")
        backend = get_backend(self.url)
        migrations = read_migrations(migrations_path)
        with backend.lock():
            backend.apply_migrations(backend.to_apply(migrations))

def init_database() -> ThreadedConnectionPool:
    if config.Config is None:
        raise RuntimeError("Config is not initialized")

    return Database().init_database()

