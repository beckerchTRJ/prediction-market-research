from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


def sqlite_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_sqlite(db_path: str | Path, schema_path: str | Path) -> None:
    schema_sql = Path(schema_path).read_text(encoding="utf-8")
    with sqlite_connection(db_path) as connection:
        connection.executescript(schema_sql)
        connection.commit()


def append_frame(db_path: str | Path, table_name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    with sqlite_connection(db_path) as connection:
        frame.to_sql(table_name, connection, if_exists="append", index=False)
        connection.commit()


def query_frame(db_path: str | Path, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with sqlite_connection(db_path) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def insert_record(db_path: str | Path, table_name: str, record: dict[str, Any]) -> int:
    columns = list(record.keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    values = [record[column] for column in columns]
    sql = f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})"
    with sqlite_connection(db_path) as connection:
        cursor = connection.execute(sql, values)
        connection.commit()
        return int(cursor.lastrowid)


def upsert_frame(
    db_path: str | Path,
    table_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if frame.empty:
        return
    where_sql = " AND ".join(f"{column} = ?" for column in key_columns)
    with sqlite_connection(db_path) as connection:
        for _, row in frame[key_columns].iterrows():
            connection.execute(
                f"DELETE FROM {table_name} WHERE {where_sql}",
                tuple(row[column] for column in key_columns),
            )
        frame.to_sql(table_name, connection, if_exists="append", index=False)
        connection.commit()

