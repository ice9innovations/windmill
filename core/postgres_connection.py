#!/usr/bin/env python3
"""
Shared PostgreSQL connection primitives for Windmill workers.

The goal is to make connection policy explicit and centralized:

- one place for psycopg2 connect kwargs
- one place for reconnect/close behavior
- one reusable class for workers that need their own managed connections
"""

from dataclasses import dataclass
from typing import Optional

import psycopg2


@dataclass(frozen=True)
class PostgresConnectionConfig:
    host: str
    database: str
    user: str
    password: str
    sslmode: Optional[str] = None
    connect_timeout: int = 10
    keepalives: int = 1
    keepalives_idle: int = 30
    keepalives_interval: int = 10
    keepalives_count: int = 3

    def connect_kwargs(self) -> dict:
        kwargs = {
            'host': self.host,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'connect_timeout': self.connect_timeout,
            'keepalives': self.keepalives,
            'keepalives_idle': self.keepalives_idle,
            'keepalives_interval': self.keepalives_interval,
            'keepalives_count': self.keepalives_count,
        }
        if self.sslmode:
            kwargs['sslmode'] = self.sslmode
        return kwargs


def create_connection(config: PostgresConnectionConfig, autocommit: bool = True):
    conn = psycopg2.connect(**config.connect_kwargs())
    conn.autocommit = autocommit
    return conn


def close_quietly(resource):
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        pass


def rollback_quietly(conn) -> bool:
    if conn is None or getattr(conn, 'closed', 1) != 0:
        return False
    try:
        conn.rollback()
        return True
    except Exception:
        return False


def commit_if_needed(conn, *, force: bool = False) -> bool:
    if conn is None or getattr(conn, 'closed', 1) != 0:
        return False
    if not force and getattr(conn, 'autocommit', False):
        return False
    conn.commit()
    return True


class ManagedPostgresConnection:
    """Small lifecycle wrapper around a psycopg2 connection."""

    def __init__(self, config: PostgresConnectionConfig, *, autocommit: bool = True, logger=None, label: str = "database"):
        self.config = config
        self.autocommit = autocommit
        self.logger = logger
        self.label = label
        self.conn = None

    def is_open(self) -> bool:
        return self.conn is not None and self.conn.closed == 0

    def close(self):
        if self.conn is None:
            return
        close_quietly(self.conn)
        self.conn = None

    def connect(self):
        self.close()
        self.conn = create_connection(self.config, autocommit=self.autocommit)
        return self.conn

    def reconnect(self):
        return self.connect()
