Ngày 1 – Hiểu dữ liệu & dựng pipeline cơ bản
Cùng nhau (30–60 phút đầu ngày)

Chốt:

Cột target (fraud), cột time (thời gian giao dịch).

Mục tiêu: flag 5% giao dịch, tối ưu Recall/Precision.

N1

Khám phá dữ liệu (EDA nhanh)

Tỷ lệ fraud tổng thể, theo thời gian (theo ngày/tháng).

train.shape, test.shape.

train.dtypes để xem kiểu dữ liệu từng cột.

Missing & chất lượng feature

Tính missing_rate cho 141 cột, sort giảm dần.

Tìm cột:

Toàn 1 giá trị → ghi lại để xem xét drop.

Numeric bị lưu dạng string → note để convert.

Output: 1 notebook 01_eda_overview.ipynb + 1 bảng nhỏ:
danh sách cột, %missing, kiểu dữ liệu.

N2

Setup code & cấu trúc

Tạo repo: src/, notebooks/, data/, models/, reports/.

Viết:

src/load_data.py → hàm đọc train/test.

src/split_time.py → hàm chia train_internal/val theo time (80/20).

Baseline XGBoost cực đơn giản

Lấy tạm vài feature numeric “sạch” nhất (N1 chỉ ra, ví dụ 10 cột).

Pipeline:

Numeric: SimpleImputer(median)

Train XGBoost baseline.

Tính:

AUC trên validation.

Chọn threshold = quantile 95% của score → tính Recall@5%, Precision@5%.

Output: script/notebook 02_baseline_xgb.ipynb chạy end-to-end.

Ngày 2 – Xử lý missing chuẩn + FE tầng 2
N1

Thiết kế cách xử lý missing cho 141 feature

Numeric:

Dùng median + flag missing (imputer indicator).

Categorical:

Điền "MISSING" hoặc mode, sau đó OneHot.

Clean & chuẩn hoá feature

Drop cột:

all-NaN,

constant,

ID vô nghĩa (nếu có).

Convert các cột numeric đang là string.

Tạo feature tầng 2 từ 141 feature có sẵn

Ratio:

user_freq / machine_freq

user_amount_mean / merchant_amount_mean

Deviation:

user_amount_std / (user_amount_mean + 1)

Log-transform:

log(1 + x) cho các count/frequency rất lệch.

Output: src/features.py với hàm
build_Xy_v1(df) -> X, y (141 feature đã clean + một số feature tầng 2).

N2

Nối pipeline với build_Xy_v1

load → build_Xy_v1(train) → split time-based → X_train, X_val, y_train, y_val.

Xây dựng ColumnTransformer chuẩn

Numeric:

SimpleImputer(strategy="median", add_indicator=True).

Categorical:

SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore").

Train XGBoost v1 nghiêm túc

Tham số vừa phải:

n_estimators ~ 300, max_depth ~ 5–7, learning_rate ~ 0.05.

Đánh giá trên validation:

AUC,

Recall@5% & Precision@5% (dùng hàm riêng).

Output: src/train_v1.py (hoặc notebook 03_xgb_v1.ipynb) + file report ngắn.

Ngày 3 – Chọn feature quan trọng & hiểu tác động
N1

Phân tích tương quan & trùng lặp

Tính correlation giữa các feature numeric.

Ghi lại:

nhóm feature corr > 0.98 → đề xuất drop bớt.

Review importance cùng N2

Nhìn top feature quan trọng từ XGBoost:

Có feature nào không hợp logic nghiệp vụ → nghi ngờ leakage.

Có nhóm feature nào “na ná nhau” → thống nhất bỏ bớt.

Cập nhật hàm build_Xy_final

Tạo build_Xy_final(df):

Dùng tập feature đã clean + chọn lọc (vd: 60–100 cột).

Đảm bảo build_Xy_final dùng chung cho train & test.

N2

Feature importance & SHAP

Từ model v1 (ngày 2):

Lấy feature importance (gain) từ XGBoost.

Tính SHAP:

summary_plot → top feature toàn cục.

dependence_plot cho 3–5 feature chính:

ví dụ: một feature user*, một merchant*, một machine\_, một tần suất…

Chọn vài case fraud → SHAP local để giải thích.

Train model v2 với build_Xy_final

Update pipeline dùng build_Xy_final.

Train lại XGBoost với param tốt nhất từ hôm trước.

Đánh giá lại AUC, Recall@5%, Precision@5%.

Output:

reports/feature_importance.md (ảnh FI, SHAP + nhận xét).

model_v2.pkl (model ứng viên).

Ngày 4 – Train final model trên full train & chạy test
N1

Check build_Xy_final trên cả train & test

Chạy:

X_train_full, y_train_full = build_Xy_final(train)

X_test = build_Xy_final(test)

Đảm bảo:

Không lỗi kiểu cột thiếu / type sai.

Sanity check phân phối

So sánh phân phối 2–3 feature chính trên train vs test.

Sau khi N2 predict, kiểm tra:

Flag rate trên test ≈ 5%,

Score không bị dính hết về 0/1.

N2

Train final model

Dùng toàn bộ train (train_internal + val):

X_train_full, y_train_full.

Tham số giống model_v2.

Lưu model_final.pkl.

Chọn threshold & predict test

Threshold:

Giữ threshold 5% đã dùng trên validation (hoặc tính lại rõ ràng từ train_full nếu bạn có quy ước).

Predict

fraud_proba = model_final.predict_proba(X_test)[:,1]

fraud_flag = (fraud_proba >= threshold)

Xuất file:

id (nếu có), fraud_proba, fraud_flag → fraud_submission.csv.

Báo cáo nhanh

1 file .md / slide:

Mục tiêu (flag 5%),

Mô tả data (141 feature thống kê),

Cách xử lý missing & FE,

Model & param chính,

AUC + Recall/Precision@5% trên validation,

Top feature ảnh hưởng (từ SHAP).
`