from __future__ import annotations
from psycopg2.pool import ThreadedConnectionPool
from repository.tx import Tx

class DataStore:

    def __init__(self, pool: ThreadedConnectionPool) -> None:
        self._pool = pool

    def query(self, fn):
        conn = self._pool.getconn()
        try:
            return fn(Tx(conn))
        finally:
            self._pool.putconn(conn)
    

    def atomic(self, fn):
        conn = self._pool.getconn()      
        try:
            tx = Tx(conn) 
            result = fn(tx)               
            conn.commit()                
            return result
        except Exception:
            conn.rollback()              
            raise
        finally:
            self._pool.putconn(conn)     

