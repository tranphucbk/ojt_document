Ngày 1 – Hiểu dữ liệu & dựng pipeline cơ bản
Người A – Data & Feature

Đọc train & test:

Kiểm tra số dòng, số cột, kiểu dữ liệu.

Xác định cột: target (fraud), time (thời gian giao dịch).

EDA nhanh 141 feature:

Tỷ lệ fraud tổng thể và theo thời gian (theo ngày/tháng).

Missing rate của từng cột → list top cột thiếu nhiều.

Tìm cột:

Toàn 1 giá trị (constant) → note để drop.

Có format sai (numeric mà lưu dạng string) → note để convert.

Phân loại feature:

Numeric vs categorical.

Theo nhóm: user_xxx, merchant_xxx, machine_xxx, frequency_xxx,…

Người B – Model & Eval

Setup repo/code:

Hàm load_data() đọc train/test.

Hàm time_based_split(train, ratio=0.8) → chia train_internal / validation theo thời gian.

Dựng pipeline đơn giản:

Dùng vài feature numeric dễ hiểu (vd: 5–10 cột).

Dùng ColumnTransformer:

numeric → SimpleImputer(median)

categorical → SimpleImputer(most_frequent) + OneHotEncoder

Train XGBoost baseline.

Đánh giá sơ bộ trên validation:

Tính AUC.

Lấy quantile 95% của score để flag 5% → tính Recall@5%, Precision@5%.

Ghi lại kết quả baseline (file markdown nhỏ).

Kết quả cuối ngày 1:
✔ Cả team hiểu cấu trúc 141 feature, vấn đề missing.
✔ Có pipeline chạy được từ train → split → train XGBoost → metric.

Ngày 2 – Xử lý missing chuẩn & FE tầng 2 từ 141 feature
Người A – Data & Feature

Thiết kế & implement chiến lược xử lý missing:

Numeric:

SimpleImputer(strategy="median", add_indicator=True)
→ điền median + sinh cột flag “đã từng thiếu”.

Categorical:

Điền "MISSING" hoặc mode, sau đó OneHot.

Loại bỏ cột:

all-NaN, constant, ID vô nghĩa.

Tạo feature tầng 2 từ 141 cột:

Ratio & deviation:

user_freq / machine_freq

user_amount_mean / merchant_amount_mean

user_amount_std / (user_amount_mean + 1),…

Log-transform cho các feature rất lệch: log(1 + x) cho count/freq.

Viết hàm build_Xy_v2(df):

Nhận df (train/test),

Trả về X (141 + feature mới), y (nếu có).

Người B – Model & Eval

Cập nhật pipeline để dùng build_Xy_v2:

load → build_Xy_v2 → split time-based → train XGBoost.

Train 1–2 model XGBoost với param đơn giản:

So sánh với baseline ngày 1:

AUC,

Recall@5%, Precision@5%.

Ghi lại thực nghiệm exp_v2:

FE v2 dùng những nhóm feature nào,

Metric có cải thiện không.

Kết quả cuối ngày 2:
✔ Bộ X, y “chuẩn” với xử lý missing rõ ràng.
✔ Đã có thêm một lớp feature tầng 2 và model tốt hơn baseline.

Ngày 3 – Chọn feature quan trọng & hiểu tác động (XGBoost + SHAP)
Người A – Data & Feature

Phân tích tương quan & redundancy:

Tính correlation giữa các feature numeric → tìm nhóm feature trùng lặp (corr rất cao).

Note những cột có thể drop bớt cho gọn.

Cùng B xem output feature importance:

Check xem các feature top có hợp lý về nghiệp vụ không.

Phát hiện khả năng data leakage (nếu có cột “quá mạnh” bất thường).

Cập nhật lại build_Xy_final(df):

Loại bớt feature rác/trùng,

Giữ lại danh sách feature “final” (vd: 60–100 cột).

Người B – Model & Eval

Train XGBoost với build_Xy_final:

Dùng param đã tương đối tốt ngày 2.

Lưu model model_candidate.

Tính feature importance:

importance_type="gain" từ XGBoost.

(Nếu kịp) thêm permutation importance trên validation.

Dùng SHAP cho model_candidate:

summary_plot → xem top feature toàn cục (user_xxx, merchant_xxx, machine_xxx,…).

dependence_plot cho 3–5 feature quan trọng nhất:

Ví dụ: user_txn_freq, merchant_fraud_rate, machine_txn_std, log_user_amount_mean…

Lấy vài case fraud điển hình → xem SHAP local explanation (vì sao bị flag).

Kết quả cuối ngày 3:
✔ Có danh sách feature “final” đã được kiểm qua nghiệp vụ.
✔ Hiểu được feature nào mạnh nhất và tác động theo hướng nào.
✔ Có model_candidate với metric được xác nhận.

Ngày 4 – Train final model trên full train & chạy test + báo cáo
Người A – Data & Feature

Đảm bảo build_Xy_final chạy được cho cả train & test:

Không lỗi kiểu: cột thiếu, type sai.

Chạy sanity check:

Phân phối một số feature chính trên train vs test (không quá lệch).

Sau khi B train xong:

Kiểm tra phân phối fraud_proba và fraud_flag trên test:

Flag rate trên test ~ 5% chưa.

Score có tập trung hết 0 hoặc 1 hay phân bố hợp lý.

Người B – Model & Eval

Train final XGBoost trên toàn bộ train (train_internal + val):

Dùng build_Xy_final(train).

Dùng param đã chốt.

Dùng threshold 5% rút ra từ validation ngày 3 (hoặc tính lại rõ ràng từ full train nếu bạn có quy ước riêng).

Dự đoán trên test:

fraud_proba = model_final.predict_proba(X_test)[:,1]

fraud_flag = (fraud_proba >= threshold)

Xuất file kết quả:

Gồm id (nếu có), fraud_proba, fraud_flag.

Viết báo cáo ngắn:

Mục tiêu: flag 5% giao dịch.

Data: 141 feature thống kê (user, merchant, machine,…).

Các bước:

Xử lý missing, FE tầng 2, feature selection.

Model:

XGBoost, hyperparameter chính.

Kết quả trên validation:

AUC,

Recall@5%, Precision@5%.

Giải thích:

Top feature từ SHAP + ý nghĩa.