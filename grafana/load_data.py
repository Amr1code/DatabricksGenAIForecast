"""
Load the two monitoring CSVs (exported by notebooks/07_model_monitoring.ipynb)
into Postgres so Grafana can query them.

Runs inside the `loader` service of docker-compose.yml, or locally:
    PGHOST=localhost PGUSER=monitor PGPASSWORD=monitor PGDATABASE=monitoring \
    DATA_DIR=./data python load_data.py

Safe to re-run: each table is dropped and rebuilt from the CSV, so the database
always mirrors the files in DATA_DIR.
"""

import csv
import os
import sys
import time

import psycopg2

DATA_DIR = os.environ.get("DATA_DIR", "./data")

# table -> (csv file, column definitions in the order the notebook exports them)
TABLES = {
    "model_metrics_history": (
        "model_metrics_history.csv",
        [
            ("run_id", "TEXT PRIMARY KEY"),
            ("run_name", "TEXT"),
            ("model_type", "TEXT NOT NULL"),
            ("run_timestamp", "TIMESTAMPTZ NOT NULL"),
            ("mape", "DOUBLE PRECISION"),
            ("rmse", "DOUBLE PRECISION"),
            ("is_simulated", "BOOLEAN"),
            ("experiment_name", "TEXT"),
            ("ingested_at", "TIMESTAMPTZ"),
        ],
    ),
    "model_drift_status": (
        "model_drift_status.csv",
        [
            ("run_id", "TEXT PRIMARY KEY"),
            ("run_name", "TEXT"),
            ("model_type", "TEXT NOT NULL"),
            ("run_timestamp", "TIMESTAMPTZ NOT NULL"),
            ("mape", "DOUBLE PRECISION"),
            ("rmse", "DOUBLE PRECISION"),
            ("is_simulated", "BOOLEAN"),
            ("baseline_run_id", "TEXT"),
            ("baseline_mape", "DOUBLE PRECISION"),
            ("mape_change_pct", "DOUBLE PRECISION"),
            ("drift_threshold", "DOUBLE PRECISION"),
            ("drift_flag", "BOOLEAN"),
            ("checked_at", "TIMESTAMPTZ"),
        ],
    ),
}


def connect(retries=10, delay=3):
    """Connect using the standard PG* environment variables, retrying briefly."""
    for attempt in range(1, retries + 1):
        try:
            return psycopg2.connect()
        except psycopg2.OperationalError as exc:
            if attempt == retries:
                raise
            print(f"Postgres not ready ({exc.__class__.__name__}), retry {attempt}/{retries}...")
            time.sleep(delay)


def read_header(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), None)
    if not header:
        sys.exit(f"ERROR: {path} is empty.")
    return [h.strip() for h in header]


def load_table(cur, table, filename, columns):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        sys.exit(
            f"ERROR: {path} not found. Export the CSVs from notebook 07 "
            f"(Step 4) and put them in the data/ folder next to docker-compose.yml."
        )

    header = read_header(path)
    expected = [name for name, _ in columns]
    missing = [c for c in expected if c not in header]
    if missing:
        sys.exit(f"ERROR: {filename} is missing column(s) {missing}. Found: {header}")

    ddl = ", ".join(f"{name} {ctype}" for name, ctype in columns)
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"CREATE TABLE {table} ({ddl})")

    # COPY maps columns by position, so require the exact column order the
    # notebook exports instead of risking values landing in the wrong column.
    if header != expected:
        sys.exit(
            f"ERROR: {filename} columns {header} do not match the expected "
            f"{expected}. Re-export the CSV from notebook 07."
        )

    with open(path, newline="", encoding="utf-8-sig") as f:
        cur.copy_expert(
            f"COPY {table} ({', '.join(expected)}) FROM STDIN WITH (FORMAT csv, HEADER true)",
            f,
        )
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"{table}: loaded {cur.fetchone()[0]} rows from {filename}")


def main():
    conn = connect()
    try:
        with conn, conn.cursor() as cur:  # one transaction: all tables or none
            cur.execute("SET TIME ZONE 'UTC'")
            for table, (filename, columns) in TABLES.items():
                load_table(cur, table, filename, columns)
    finally:
        conn.close()
    print("Done.")


if __name__ == "__main__":
    main()