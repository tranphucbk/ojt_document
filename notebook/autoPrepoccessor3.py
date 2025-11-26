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
- Xử lý outlier cho numeric (clip theo IQR hoặc z-score).
- Loại bớt feature numeric có correlation quá cao.
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
    - Xử lý outlier cho numeric (IQR / Z-score)
    - Loại bỏ feature numeric có correlation cao
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
        handle_outliers: bool = True,
        outlier_method: str = "iqr",     # "iqr" hoặc "zscore"
        outlier_iqr_factor: float = 1.5,
        outlier_z_thresh: float = 3.0,
        drop_high_corr: bool = True,
        corr_threshold: float = 0.95,
    ):
        self.cat_threshold = cat_threshold
        self.one_hot_threshold = one_hot_threshold
        self.max_missing_ratio = max_missing_ratio

        self.add_missing_flags = add_missing_flags
        self.missing_flag_suffix = missing_flag_suffix

        self.scale_numeric = scale_numeric
        self.scale_method = scale_method

        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_iqr_factor = outlier_iqr_factor
        self.outlier_z_thresh = outlier_z_thresh

        self.drop_high_corr = drop_high_corr
        self.corr_threshold = corr_threshold

        # Lưu cấu hình cột
        self.columns_: Dict[str, ColumnConfig] = {}
        self.fill_values_: Dict[str, Any] = {}

        self.label_maps_: Dict[str, Dict[Any, int]] = {}
        self.one_hot_categories_: Dict[str, List[Any]] = {}

        self.missing_ratios_: Dict[str, float] = {}
        self.dropped_columns_: List[str] = []          # drop do missing
        self.dropped_corr_features_: List[str] = []    # drop do correlation
        self.output_columns_: List[str] = []

        # Thống kê để scale numeric
        # standard: {"mean": ..., "std": ...}
        # minmax:   {"min": ..., "max": ...}
        self.scaler_stats_: Dict[str, Dict[str, float]] = {}

        # Giới hạn outlier cho numeric
        # {"col": {"lower": ..., "upper": ...}}
        self.outlier_bounds_: Dict[str, Dict[str, float]] = {}

    # ---------- Helpers ----------
    def _infer_type(self, s: pd.Series) -> Tuple[bool, str]:
        dtype = str(s.dtype)
        u = s.nunique(dropna=True)

        if dtype in ["object", "category", "bool"]:
            return True, dtype
        if u <= self.cat_threshold:
            return True, dtype
        return False, dtype  # numeric

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
        numeric_cols: List[str] = []

        # --- 1) Thống kê theo cột, bỏ cột missing nhiều, xác định loại, fill, encoder ---
        for col in df.columns:
            s = df[col]
            mratio = s.isna().sum() / total if total else 0.0
            self.missing_ratios_[col] = mratio

            # Drop do missing nhiều
            if mratio > self.max_missing_ratio:
                self.dropped_columns_.append(col)
                continue

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

            # Fill values
            if is_cat:
                fill = s.mode().iloc[0] if not s.dropna().empty else None
            else:
                fill = float(s.mean()) if not s.dropna().empty else 0.0
                numeric_cols.append(col)
            self.fill_values_[col] = fill

            # Encoders
            if encoding == "label":
                cats = sorted(s.dropna().unique().tolist())
                self.label_maps_[col] = {v: i for i, v in enumerate(cats)}
            elif encoding == "onehot":
                self.one_hot_categories_[col] = sorted(s.dropna().unique().tolist())

        # --- 2) Tính outlier bounds + scaler stats cho numeric ---
        for col in numeric_cols:
            s = df[col]
            clean = s.dropna()
            if clean.empty:
                continue

            # 2.1 Outlier bounds
            if self.handle_outliers:
                if self.outlier_method == "iqr":
                    q1 = float(clean.quantile(0.25))
                    q3 = float(clean.quantile(0.75))
                    iqr = q3 - q1
                    if iqr == 0:
                        lower = q1
                        upper = q3
                    else:
                        lower = q1 - self.outlier_iqr_factor * iqr
                        upper = q3 + self.outlier_iqr_factor * iqr
                else:  # zscore
                    mean = float(clean.mean())
                    std = float(clean.std()) or 1.0
                    lower = mean - self.outlier_z_thresh * std
                    upper = mean + self.outlier_z_thresh * std

                self.outlier_bounds_[col] = {"lower": lower, "upper": upper}
                clean_for_stats = clean.clip(lower=lower, upper=upper)
            else:
                clean_for_stats = clean

            # 2.2 Scale stats
            if self.scale_numeric:
                if self.scale_method == "standard":
                    mean = float(clean_for_stats.mean())
                    std = float(clean_for_stats.std()) or 1.0
                    self.scaler_stats_[col] = {"mean": mean, "std": std}
                elif self.scale_method == "minmax":
                    minv = float(clean_for_stats.min())
                    maxv = float(clean_for_stats.max())
                    if maxv == minv:
                        maxv = minv + 1e-9
                    self.scaler_stats_[col] = {"min": minv, "max": maxv}

        # --- 3) Loại bỏ feature tương quan cao ---
        if self.drop_high_corr and numeric_cols:
            num_df = df[numeric_cols].copy()

            # Fill + clip trước khi tính corr
            for col in numeric_cols:
                fv = self.fill_values_.get(col, 0.0)
                num_df[col] = num_df[col].fillna(fv)

                if self.handle_outliers and col in self.outlier_bounds_:
                    b = self.outlier_bounds_[col]
                    num_df[col] = num_df[col].clip(lower=b["lower"], upper=b["upper"])

            corr = num_df.corr().abs()

            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

            to_drop = set()
            for i in upper.columns:
                for j in upper.index:
                    val = upper.loc[j, i]
                    if pd.isna(val):
                        continue
                    if val >= self.corr_threshold:
                        mr_i = self.missing_ratios_.get(i, 0.0)
                        mr_j = self.missing_ratios_.get(j, 0.0)
                        drop_col = i if mr_i >= mr_j else j
                        if drop_col in numeric_cols:
                            to_drop.add(drop_col)

            for col in to_drop:
                if col in self.columns_:
                    self.dropped_corr_features_.append(col)
                    self.columns_.pop(col, None)
                    self.fill_values_.pop(col, None)
                    self.scaler_stats_.pop(col, None)
                    self.outlier_bounds_.pop(col, None)

        # --- 4) Xác định thứ tự cột output ---
        self.output_columns_ = self._compute_output_columns()
        return self

    def _compute_output_columns(self) -> List[str]:
        cols: List[str] = []

        for col, cfg in self.columns_.items():
            if cfg.is_categorical and cfg.encoding == "onehot":
                for c in self.one_hot_categories_.get(col, []):
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
            raise RuntimeError("fit() must be gọi trước transform().")

        orig = df.copy()
        work = df.copy()

        # 1) Xoá cột đã drop (do missing hoặc correlation)
        for col in self.dropped_columns_ + self.dropped_corr_features_:
            if col in work.columns:
                work = work.drop(columns=[col])

        # 2) Tạo missing flag
        flags: Dict[str, pd.Series] = {}
        if self.add_missing_flags:
            for col in self.columns_:
                if col in orig.columns:
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

        # 4) Clip outlier cho numeric (nếu bật)
        if self.handle_outliers:
            for col, bounds in self.outlier_bounds_.items():
                if col in work.columns:
                    work[col] = pd.to_numeric(work[col], errors="coerce").clip(
                        lower=bounds["lower"], upper=bounds["upper"]
                    )

        # 5) Encode + scale
        dfs: List[pd.DataFrame] = []

        for col, cfg in self.columns_.items():
            s = work[col]

            if cfg.is_categorical and cfg.encoding == "onehot":
                temp = {}
                for c in self.one_hot_categories_.get(col, []):
                    temp[f"{col}__{c}"] = (s == c).astype(int)
                dfs.append(pd.DataFrame(temp, index=work.index))

            elif cfg.is_categorical and cfg.encoding == "label":
                mapping = self.label_maps_.get(col, {})
                dfs.append(s.map(mapping).fillna(-1).astype(int).to_frame(col))

            else:
                s_num = pd.to_numeric(s, errors="coerce")

                if self.scale_numeric and col in self.scaler_stats_:
                    stats = self.scaler_stats_[col]
                    if self.scale_method == "standard":
                        s_num = (s_num - stats["mean"]) / stats["std"]
                    elif self.scale_method == "minmax":
                        s_num = (s_num - stats["min"]) / (stats["max"] - stats["min"])

                dfs.append(s_num.to_frame(col))

        # 6) Thêm cột flag vào cuối
        if flags:
            dfs.append(pd.DataFrame(flags, index=orig.index))

        result = pd.concat(dfs, axis=1)
        result = result.reindex(columns=self.output_columns_)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # =================================================================
    #                          UTILITIES
    # =================================================================
    def get_column_configs(self) -> pd.DataFrame:
        rows = []

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

        for col in self.dropped_columns_ + self.dropped_corr_features_:
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
        In ra toàn bộ cột được xác định là categorical
        """
        print("====== CÁC CỘT CATEGORICAL ======")
        any_cat = False
        for col, cfg in self.columns_.items():
            if cfg.is_categorical:
                any_cat = True
                print(
                    f"- {col:<25} | encoding = {str(cfg.encoding):<8} | unique = {cfg.n_unique}"
                )
        if not any_cat:
            print("(Không có cột categorical nào được nhận diện)")

    def print_fill_values(self):
        """
        In ra giá trị fill của từng cột
        """
        print("====== GIÁ TRỊ FILL CHO CÁC CỘT ======")
        for col, cfg in self.columns_.items():
            fv = self.fill_values_.get(col, None)
            kind = "categorical" if cfg.is_categorical else "numeric"
            print(f"- {col:<25} | {kind:<11} | fill = {fv}")


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

        result["processed"]      -> X sau khi preprocess
        result["summary"]        -> bảng missing/unique
        result["configs"]        -> cấu hình cột
        result["dropped"]        -> list cột drop (missing + corr)
        result["distributions"]  -> dict phân phối
        result["preprocessor"]   -> chính object AutoPreprocessor
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
        "dropped": ap.dropped_columns_ + ap.dropped_corr_features_,
        "distributions": dists,
        "preprocessor": ap,
    }
