Đã thực hiện:

- Xử lý dữ liệu thiếu. Loại bỏ các cột dữ liệu không đóng góp nhiều giá trị (dự trên phần phối)
- Thêm các cột flag để đánh dấu giá trị thiếu.
- Loại bỏ các feature có correlaiton cao với nhau hoặc quá thấp với target.
- Thử mô hình mới xgboost, lightgbm, logistic. Thử tuning với optuna
- Thử nghiệm ensamble
- Giới hạn để 5% dự đoán để giảm bảo tỷ lệ.

- Kết quả đặt được : Kết quả toàn trước (khi chỉ lọc tiến hành cơ bản) với random forest: kết quả thu được là .....

Tuần này: khi sử dụng mô hình phức tạp hơn cùng với thêm luwognj feature mới : kết quả bài toán đang có dấy hiệu giảm -> cần phải giảm số lượng feature .

Công việc tuần tới.

- Tiếp tục thử nghiệm mô hình với optuna.
- Đặt thêm các ngưỡng và xem xét lại phân phối để tìm ra các feature mới, cũng như giảm lượng feature mô hình sử dụng.
