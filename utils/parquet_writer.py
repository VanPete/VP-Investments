import os
from typing import Optional

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAS_PARQUET = True
except Exception:  # pragma: no cover
    _HAS_PARQUET = False

import pandas as pd


def write_both(df: pd.DataFrame, csv_path: str) -> Optional[str]:
    """Write CSV as usual and Parquet alongside if pyarrow is available.
    Returns the parquet path if written.
    """
    # Write CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    if not _HAS_PARQUET:
        return None

    parquet_path = os.path.splitext(csv_path)[0] + ".parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression="zstd")
    return parquet_path
