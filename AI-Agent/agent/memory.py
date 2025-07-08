from typing import Dict
import pandas as pd

class DataFrameStore:
    """Singleton lưu DataFrame theo key (mã CK)."""
    _store: Dict[str, pd.DataFrame] = {}

    @classmethod
    def save(cls, key: str, df: pd.DataFrame) -> None:
        cls._store[key] = df

    @classmethod
    def get(cls, key: str) -> pd.DataFrame | None:
        return cls._store.get(key)

    @classmethod
    def keys(cls):
        return list(cls._store.keys())

    @classmethod
    def clear(cls):
        cls._store.clear()
