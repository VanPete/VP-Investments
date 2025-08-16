import os
from typing import List, Optional

try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None


def list_runs(outputs_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))])


def query_final_analysis(outputs_dir: str, sql: str, limit: Optional[int] = 1000):
    """Run a DuckDB SQL query over the latest Final Analysis parquet/CSV file.
    Returns a list of dict rows, or None if duckdb isn't available.
    """
    if duckdb is None:
        return None

    latest = None
    for run in reversed(list_runs(outputs_dir)):
        # Prefer parquet
        pq = os.path.join(outputs_dir, run, "Final Analysis.parquet")
        csv = os.path.join(outputs_dir, run, "Final Analysis.csv")
        if os.path.exists(pq) or os.path.exists(csv):
            latest = pq if os.path.exists(pq) else csv
            break
    if not latest:
        return []

    con = duckdb.connect()
    try:
        # Create a view over the file
        if latest.endswith(".parquet"):
            con.execute("CREATE OR REPLACE VIEW final AS SELECT * FROM read_parquet(?);", [latest])
        else:
            con.execute("CREATE OR REPLACE VIEW final AS SELECT * FROM read_csv_auto(?);", [latest])
        q = sql if limit is None else f"{sql} LIMIT {int(limit)}"
        res = con.execute(q).fetchall()
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, row)) for row in res]
    finally:
        con.close()
