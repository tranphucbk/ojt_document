"""
autoPreprocessor2.py

Thư viện dùng:
- pandas
- numpy
- dataclasses
- typing

Chức năng chính:
- Thống kê missing, unique, distribution.
- Tự phân loại numeric / categorical.
- Categorical: one-hot hoặc label encoding.
- Fill missing:
    + Categorical -> mode
    + Numeric -> mean
- Drop cột nếu tỷ lệ missing > max_missing_ratio.
- Thêm cột flag missing (0/1) cho từng cột.
- Scale numeric (standardize hoặc min-max).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple


# =====================================================================
#                            COLUMN CONFIG
# =====================================================================
@dataclass
class ColumnConfig:
    name: str
    dtype: str
    is_categorical: bool
    n_unique: int
    missing_ratio: float
    encoding: Optional[str] = None   # "onehot" / "label" / None


# =====================================================================
#                           DATA PROFILER
# =====================================================================
class DataProfiler:
    """
    Thống kê:
    - Missing %
    - Số unique
    - Distribution (tỷ trọng)
    """

    def basic_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        total = len(df)
        rows = []

        for col in df.columns:
            s = df[col]
            missing = s.isna().sum()
            unique = s.nunique(dropna=True)

            rows.append(
                {
                    "column": col,
                    "dtype": str(s.dtype),
                    "missing_count": int(missing),
                    "missing_pct": float(missing) * 100 / total if total > 0 else np.nan,
                    "unique": int(unique),
                }
            )

        return pd.DataFrame(rows)

    def distributions(self, df: pd.DataFrame, normalize: bool = True) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}

        for col in df.columns:
            s = df[col]

            if pd.api.types.is_numeric_dtype(s):
                if s.notna().sum() == 0:
                    out[col] = pd.Series(dtype=float)
                else:
                    try:
                        bins = pd.cut(s, bins=10)
                        out[col] = bins.value_counts(normalize=normalize).sort_index()
                    except ValueError:
                        out[col] = s.value_counts(normalize=normalize)
            else:
                out[col] = s.value_counts(normalize=normalize)

        return out


# =====================================================================
#                        AUTO PREPROCESSOR
# =====================================================================
class AutoPreprocessor:
    """
    - Xác định cột categorical / numeric
    - One-hot vs Label encoding
    - Fill dữ liệu thiếu
    - Drop cột thiếu quá nhiều
    - Tạo missing_flag
    - Scale numeric (standardize hoặc min-max)
    """

    def __init__(
        self,
        cat_threshold: int = 20,
        one_hot_threshold: int = 10,
        max_missing_ratio: float = 0.8,
        add_missing_flags: bool = True,
        missing_flag_suffix: str = "_missing",
        scale_numeric: bool = True,
        scale_method: str = "standard",  # "standard" hoặc "minmax"
    ):
        self.cat_threshold = cat_threshold
        self.one_hot_threshold = one_hot_threshold
        self.max_missing_ratio = max_missing_ratio

        self.add_missing_flags = add_missing_flags
        self.missing_flag_suffix = missing_flag_suffix

        self.scale_numeric = scale_numeric
        self.scale_method = scale_method

        # Lưu cấu hình
        self.columns_: Dict[str, ColumnConfig] = {}
        self.fill_values_: Dict[str, Any] = {}

        self.label_maps_: Dict[str, Dict[Any, int]] = {}
        self.one_hot_categories_: Dict[str, List[Any]] = {}

        self.missing_ratios_: Dict[str, float] = {}
        self.dropped_columns_: List[str] = []
        self.output_columns_: List[str] = []

        # Thống kê để scale numeric
        # standard: {"mean": ..., "std": ...}
        # minmax:   {"min": ..., "max": ...}
        self.scaler_stats_: Dict[str, Dict[str, float]] = {}

    # ---------- Helpers ----------
    def _infer_type(self, s: pd.Series) -> Tuple[bool, str]:
        dtype = str(s.dtype)
        u = s.nunique(dropna=True)

        if dtype in ["object", "category", "bool"]:
            return True, dtype
        if u <= self.cat_threshold:
            return True, dtype
        return False, dtype

    def _choose_encoding(self, is_cat: bool, u: int) -> Optional[str]:
        if not is_cat:
            return None
        if u <= self.one_hot_threshold:
            return "onehot"
        return "label"

    # =================================================================
    #                               FIT
    # =================================================================
    def fit(self, df: pd.DataFrame) -> "AutoPreprocessor":
        total = len(df)

        for col in df.columns:
            s = df[col]
            mratio = s.isna().sum() / total if total else 0.0
            self.missing_ratios_[col] = mratio

            # 1) DROP nếu thiếu quá nhiều
            if mratio > self.max_missing_ratio:
                self.dropped_columns_.append(col)
                continue

            # 2) XÁC ĐỊNH LOẠI
            is_cat, dtype = self._infer_type(s)
            u = s.nunique(dropna=True)
            encoding = self._choose_encoding(is_cat, u)

            cfg = ColumnConfig(
                name=col,
                dtype=dtype,
                is_categorical=is_cat,
                n_unique=int(u),
                missing_ratio=float(mratio),
                encoding=encoding,
            )
            self.columns_[col] = cfg

            # 3) TÍNH GIÁ TRỊ FILL
            if is_cat:
                fill = s.mode().iloc[0] if not s.dropna().empty else None
            else:
                fill = float(s.mean()) if not s.dropna().empty else 0.0
            self.fill_values_[col] = fill

            # 4) TẠO ENCODING MAP
            if encoding == "label":
                cats = sorted(s.dropna().unique().tolist())
                self.label_maps_[col] = {v: i for i, v in enumerate(cats)}
            elif encoding == "onehot":
                self.one_hot_categories_[col] = sorted(s.dropna().unique().tolist())

            # 5) LƯU THỐNG KÊ SCALE CHO NUMERIC
            if self.scale_numeric and (not is_cat):
                clean = s.dropna()
                if not clean.empty:
                    if self.scale_method == "standard":
                        mean = float(clean.mean())
                        std = float(clean.std()) or 1.0
                        self.scaler_stats_[col] = {"mean": mean, "std": std}
                    elif self.scale_method == "minmax":
                        minv = float(clean.min())
                        maxv = float(clean.max())
                        if maxv == minv:
                            maxv = minv + 1e-9
                        self.scaler_stats_[col] = {"min": minv, "max": maxv}

        # 6) Tính danh sách cột đầu ra
        self.output_columns_ = self._compute_output_columns()
        return self

    def _compute_output_columns(self) -> List[str]:
        cols: List[str] = []

        for col, cfg in self.columns_.items():
            if cfg.encoding == "onehot":
                for c in self.one_hot_categories_[col]:
                    cols.append(f"{col}__{c}")
            else:
                cols.append(col)

        if self.add_missing_flags:
            for col in self.columns_:
                cols.append(f"{col}{self.missing_flag_suffix}")

        return cols

    # =================================================================
    #                              TRANSFORM
    # =================================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_ and not self.dropped_columns_:
            raise RuntimeError("Bạn cần gọi fit() trước khi transform().")

        orig = df.copy()
        work = df.copy()

        # 1) Remove columns đã drop
        for col in self.dropped_columns_:
            if col in work:
                work = work.drop(columns=[col])

        # 2) Missing flag
        flags: Dict[str, pd.Series] = {}
        if self.add_missing_flags:
            for col in self.columns_:
                if col in orig:
                    flags[f"{col}{self.missing_flag_suffix}"] = orig[col].isna().astype(int)
                else:
                    flags[f"{col}{self.missing_flag_suffix}"] = pd.Series(
                        1, index=orig.index, dtype=int
                    )

        # 3) Fill missing
        for col, v in self.fill_values_.items():
            if col not in work.columns:
                work[col] = v
            work[col] = work[col].fillna(v)

        # 4) Encode + scale numeric
        dfs: List[pd.DataFrame] = []

        for col, cfg in self.columns_.items():
            s = work[col]

            if cfg.encoding == "onehot":
                temp = {}
                for c in self.one_hot_categories_[col]:
                    temp[f"{col}__{c}"] = (s == c).astype(int)
                dfs.append(pd.DataFrame(temp, index=work.index))

            elif cfg.encoding == "label":
                mapping = self.label_maps_[col]
                dfs.append(
                    s.map(mapping).fillna(-1).astype(int).to_frame(col)
                )

            else:
                # numeric: có thể scale
                s_num = pd.to_numeric(s, errors="coerce")

                if self.scale_numeric and (col in self.scaler_stats_):
                    stats = self.scaler_stats_[col]
                    if self.scale_method == "standard":
                        s_num = (s_num - stats["mean"]) / stats["std"]
                    elif self.scale_method == "minmax":
                        s_num = (s_num - stats["min"]) / (stats["max"] - stats["min"])

                dfs.append(s_num.to_frame(col))

        # 5) Add flags
        if flags:
            dfs.append(pd.DataFrame(flags, index=orig.index))

        result = pd.concat(dfs, axis=1)

        # reorder cột
        result = result.reindex(columns=self.output_columns_)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # =================================================================
    #                          UTILITIES
    # =================================================================
    def get_column_configs(self) -> pd.DataFrame:
        rows = []

        # Cột còn giữ lại sau khi fit
        for col, cfg in self.columns_.items():
            rows.append(
                {
                    "column": col,
                    "dtype": cfg.dtype,
                    "is_categorical": cfg.is_categorical,
                    "unique": cfg.n_unique,
                    "missing_ratio": cfg.missing_ratio,
                    "encoding": cfg.encoding,
                    "dropped": False,
                }
            )

        # Cột đã bị drop vì missing_ratio > max_missing_ratio
        for col in self.dropped_columns_:
            rows.append(
                {
                    "column": col,
                    "dtype": None,
                    "is_categorical": None,
                    "unique": None,
                    "missing_ratio": self.missing_ratios_.get(col, None),
                    "encoding": "DROPPED",
                    "dropped": True,
                }
            )

        return pd.DataFrame(rows)
    def print_categorical_columns(self):
        """
        In ra toàn bộ cột được xác định là categorical,
        bao gồm loại encoding (onehot/label).
        """
        print("====== CÁC CỘT CATEGORICAL ======")
        for col, cfg in self.columns_.items():
            if cfg.is_categorical:
                print(f"- {col:<25} | encoding = {cfg.encoding:<8} | unique = {cfg.n_unique}")

        if not any(cfg.is_categorical for cfg in self.columns_.values()):
            print("(Không có cột categorical nào được nhận diện)")


    def print_fill_values(self):
        """
        In ra giá trị được dùng để fill cho từng cột.
        Numeric -> mean
        Categorical -> mode
        """
        print("====== GIÁ TRỊ FILL CHO CÁC CỘT ======")
        for col, cfg in self.columns_.items():
            fv = self.fill_values_.get(col, None)
            if cfg.is_categorical:
                print(f"- {col:<25} | categorical | fill = {fv}")
            else:
                print(f"- {col:<25} | numeric     | fill = {fv}")


# =====================================================================
#                          QUICK FUNCTION
# =====================================================================
def auto_profile_and_preprocess(
    df: pd.DataFrame,
    **preprocessor_kwargs: Any,
) -> Dict[str, Any]:
    """
    Helper nhanh:
        result = auto_profile_and_preprocess(df)

        result["processed"]   -> X sau khi preprocess
        result["summary"]     -> bảng missing/unique
        result["configs"]     -> cấu hình cột
        result["dropped"]     -> list cột drop
        result["distributions"] -> dict phân phối
    """
    ap = AutoPreprocessor(**preprocessor_kwargs)
    X = ap.fit_transform(df)

    profiler = DataProfiler()
    summary = profiler.basic_summary(df)
    dists = profiler.distributions(df)

    return {
        "processed": X,
        "summary": summary,
        "configs": ap.get_column_configs(),
        "dropped": ap.dropped_columns_,
        "distributions": dists,
        "preprocessor": ap,
    }
