from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

@dataclass
class ColumnConfig:
    name: str
    dtype: str
    is_categorical: bool
    n_unique: int
    missing_ratio: float        # NEW: tỷ lệ thiếu trên cột
    encoding: Optional[str] = None  # "onehot", "label" hoặc None
class AutoPreprocessor:
    """
    Tự động:
    - Phân loại cột numeric / categorical.
    - Quyết định one-hot hay label encoding cho categorical.
    - Tính giá trị fill (mean / mode).
    - Biến đổi DataFrame: fill missing + encode.
    - MỚI:
        + Loại bỏ các cột có tỷ lệ missing > max_missing_ratio.
        + Thêm cột flag cho biết ô nào ban đầu bị missing (0/1).

    Tham số:
        cat_threshold: nếu cột có số lượng giá trị khác nhau <= cat_threshold
                       hoặc dtype = object/category -> coi là categorical.
        one_hot_threshold: nếu số lượng giá trị khác nhau <= one_hot_threshold
                           -> dùng one-hot, ngược lại -> label encoding.
        max_missing_ratio: nếu tỷ lệ missing của cột > ngưỡng này -> DROP cột.
        add_missing_flags: có tạo thêm cột flag missing hay không.
        missing_flag_suffix: hậu tố thêm vào tên cột flag.
    """

    def __init__(
        self,
        cat_threshold: int = 20,
        one_hot_threshold: int = 10,
        max_missing_ratio: float = 0.8,
        add_missing_flags: bool = True,
        missing_flag_suffix: str = "_missing",
    ):
        self.cat_threshold = cat_threshold
        self.one_hot_threshold = one_hot_threshold
        self.max_missing_ratio = max_missing_ratio
        self.add_missing_flags = add_missing_flags
        self.missing_flag_suffix = missing_flag_suffix

        # Cấu hình cho từng cột
        self.columns_: Dict[str, ColumnConfig] = {}

        # Giá trị dùng để fill
        self.fill_values_: Dict[str, Any] = {}

        # mapping cho label encoding: col -> {category: int}
        self.label_maps_: Dict[str, Dict[Any, int]] = {}

        # categories cho one-hot: col -> [category1, category2, ...]
        self.one_hot_categories_: Dict[str, List[Any]] = {}

        # Tỷ lệ missing theo cột
        self.missing_ratios_: Dict[str, float] = {}

        # Cột bị drop vì missing quá nhiều
        self.dropped_columns_: List[str] = []

        # Lưu thứ tự cột sau khi transform
        self.output_columns_: List[str] = []

    # ======= PHẦN PHÂN LOẠI CỘT =======

    def _infer_column_type(
        self, series: pd.Series
    ) -> Tuple[bool, str]:
        """
        Trả về (is_categorical, dtype_string)
        Quy tắc:
        - Nếu dtype là object / category -> categorical.
        - Nếu số lượng giá trị khác nhau <= cat_threshold -> coi là categorical.
        Ngược lại -> numeric (hoặc continuous).
        """
        dtype_str = str(series.dtype)
        n_unique = series.nunique(dropna=True)

        if dtype_str in ["object", "category", "bool"]:
            return True, dtype_str

        # Nếu số lượng giá trị khác nhau ít -> coi như categorical
        if n_unique <= self.cat_threshold:
            return True, dtype_str

        # Ngược lại -> numeric
        return False, dtype_str

    def _choose_encoding(self, is_categorical: bool, n_unique: int) -> Optional[str]:
        """
        Nếu không phải categorical -> None.
        Nếu categorical:
            - n_unique <= one_hot_threshold -> "onehot"
            - ngược lại -> "label"
        """
        if not is_categorical:
            return None

        if n_unique <= self.one_hot_threshold:
            return "onehot"
        return "label"

    # ======= FIT =======

    def fit(self, df: pd.DataFrame) -> "AutoPreprocessor":
        """
        Học cấu hình từ DataFrame:
        - Loại cột & encoding.
        - Giá trị fill (mean hoặc mode).
        - mapping cho label encoding & categories cho one-hot.
        - Drop các cột có tỷ lệ missing > max_missing_ratio.
        """
        self.columns_.clear()
        self.fill_values_.clear()
        self.label_maps_.clear()
        self.one_hot_categories_.clear()
        self.missing_ratios_.clear()
        self.dropped_columns_.clear()
        self.output_columns_.clear()

        total_rows = len(df)

        for col in df.columns:
            s = df[col]
            # Tỷ lệ missing của cột
            if total_rows > 0:
                missing_ratio = float(s.isna().sum()) / total_rows
            else:
                missing_ratio = 0.0

            self.missing_ratios_[col] = missing_ratio

            # Nếu missing quá nhiều -> drop cột
            if missing_ratio > self.max_missing_ratio:
                self.dropped_columns_.append(col)
                continue

            is_cat, dtype_str = self._infer_column_type(s)
            n_unique = s.nunique(dropna=True)
            encoding = self._choose_encoding(is_cat, n_unique)

            self.columns_[col] = ColumnConfig(
                name=col,
                dtype=dtype_str,
                is_categorical=is_cat,
                n_unique=int(n_unique),
                missing_ratio=missing_ratio,
                encoding=encoding,
            )

            # Tính fill values
            if is_cat:
                # fill bằng mode
                if s.dropna().empty:
                    fill_value = None
                else:
                    fill_value = s.mode(dropna=True).iloc[0]
                self.fill_values_[col] = fill_value
            else:
                # numeric -> fill bằng mean
                if s.dropna().empty:
                    fill_value = 0.0
                else:
                    fill_value = float(s.mean())
                self.fill_values_[col] = fill_value

            # Thiết lập mapping cho encode
            if encoding == "label":
                # Tạo mapping category -> int
                cats = s.dropna().unique()
                cats_sorted = sorted(cats.tolist())
                mapping = {cat: idx for idx, cat in enumerate(cats_sorted)}
                self.label_maps_[col] = mapping
            elif encoding == "onehot":
                cats = s.dropna().unique()
                cats_sorted = sorted(cats.tolist())
                self.one_hot_categories_[col] = cats_sorted

        # Xác định thứ tự cột đầu ra
        self.output_columns_ = self._compute_output_columns()

        return self

    def _compute_output_columns(self) -> List[str]:
        """
        Tính danh sách tên cột sau khi transform
        (bao gồm cả cột feature & cột flag missing).
        """
        output_cols: List[str] = []

        # Cột features (numeric / label / onehot)
        for col, cfg in self.columns_.items():
            if cfg.is_categorical and cfg.encoding == "onehot":
                # tạo cột col__category
                cats = self.one_hot_categories_.get(col, [])
                for cat in cats:
                    new_name = f"{col}__{cat}"
                    output_cols.append(new_name)
            else:
                # numeric hoặc categorical-label
                output_cols.append(col)

        # Cột flag missing
        if self.add_missing_flags:
            for col in self.columns_.keys():
                flag_name = f"{col}{self.missing_flag_suffix}"
                output_cols.append(flag_name)

        return output_cols

    # ======= TRANSFORM =======

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Biến đổi DataFrame:
        - Bỏ các cột đã bị đánh dấu drop vì missing quá nhiều.
        - Tạo cột flag missing (0/1) cho từng cột còn lại (dựa trên dữ liệu gốc).
        - Fill missing (categorical -> mode; numeric -> mean).
        - Encode (one-hot / label).
        - Đảm bảo cột đầu ra cố định như khi fit.
        """
        if not self.columns_ and not self.dropped_columns_:
            raise RuntimeError("Bạn cần gọi fit() trước khi transform().")

        # Giữ lại bản gốc để tạo flag missing
        orig_df = df.copy()

        # DataFrame làm việc: chỉ giữ các cột đã được giữ ở fit (không drop)
        work_df = df.copy()
        # Xóa cột đã drop (nếu vẫn còn trong dữ liệu mới)
        for col in self.dropped_columns_:
            if col in work_df.columns:
                work_df = work_df.drop(columns=[col])

        # B1: tạo cột flag missing (dựa trên orig_df, TRƯỚC khi fill)
        flag_data: Dict[str, pd.Series] = {}
        if self.add_missing_flags:
            for col in self.columns_.keys():
                if col in orig_df.columns:
                    flag_series = orig_df[col].isna().astype(int)
                else:
                    # Nếu dữ liệu mới không có cột này, coi như missing hết (1)
                    flag_series = pd.Series(1, index=orig_df.index, dtype=int)
                flag_name = f"{col}{self.missing_flag_suffix}"
                flag_data[flag_name] = flag_series

        # B2: fill missing trên work_df
        for col, fill_value in self.fill_values_.items():
            if col not in work_df.columns:
                # cột không có trong dữ liệu mới -> tạo cột toàn missing rồi fill
                work_df[col] = np.nan
            work_df[col] = work_df[col].fillna(fill_value)

        # B3: encode
        output_frames: List[pd.DataFrame] = []

        for col, cfg in self.columns_.items():
            s = work_df[col]

            if cfg.is_categorical and cfg.encoding == "onehot":
                cats = self.one_hot_categories_.get(col, [])
                # tạo frame từ 0/1
                data = {}
                for cat in cats:
                    new_name = f"{col}__{cat}"
                    data[new_name] = (s == cat).astype(int)
                out_df = pd.DataFrame(data, index=work_df.index)
                output_frames.append(out_df)

            elif cfg.is_categorical and cfg.encoding == "label":
                mapping = self.label_maps_.get(col, {})
                # giá trị không nằm trong mapping -> -1
                encoded = s.map(mapping).fillna(-1).astype(int)
                output_frames.append(encoded.to_frame(col))

            else:
                # numeric, giữ nguyên
                output_frames.append(s.to_frame(col))

        # B4: thêm các cột flag missing
        if self.add_missing_flags and flag_data:
            flags_df = pd.DataFrame(flag_data, index=orig_df.index)
            output_frames.append(flags_df)

        result = pd.concat(output_frames, axis=1)

        # B5: đảm bảo thứ tự cột theo output_columns_
        result = result.reindex(columns=self.output_columns_)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fit + transform trong 1 bước.
        """
        return self.fit(df).transform(df)

    # ======= TIỆN ÍCH =======

    def get_column_configs(self):
        rows = []

        # Cột còn giữ
        for col, cfg in self.columns_.items():
            rows.append({
                "column": col,
                "dtype": cfg.dtype,
                "is_categorical": cfg.is_categorical,
                "unique": cfg.n_unique,
                "missing_ratio": cfg.missing_ratio,
                "encoding": cfg.encoding,
                "dropped": False
            })

        # Cột bị drop
        for col in self.dropped_columns_:
            rows.append({
                "column": col,
                "dtype": None,
                "is_categorical": None,
                "unique": None,
                "missing_ratio": self.missing_ratios_.get(col, None),
                "encoding": "DROPPED",
                "dropped": True
            })

        return pd.DataFrame(rows)


    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Kết hợp:
        - summary (missing %, n_unique)
        - column configs (loại cột & encoding & missing_ratio)
        - distributions (phân bố tỷ trọng)
        """
        profiler = DataProfiler()
        summary = profiler.basic_summary(df)
        distributions = profiler.distributions(df)

        return {
            "summary": summary,
            "column_configs": self.get_column_configs(),
            "distributions": distributions,
            "dropped_columns": self.dropped_columns_,
        }
