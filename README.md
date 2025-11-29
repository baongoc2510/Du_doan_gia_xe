# ỨNG DỤNG DỰ ĐOÁN GIÁ XE MÁY CŨ & PHÁT HIỆN GIÁ BẤT THƯỜNG

Ứng dụng Streamlit cho phép:

* Dự đoán giá xe máy cũ dựa trên mô hình học máy (ví dụ Random Forest).
* Kiểm tra một tin rao có **giá bất thường** hay không (dựa trên residual / residual_z so với thống kê phân khúc).
* Lập danh sách các tin rao có giá bất thường để admin duyệt / xóa hàng loạt.

---

## Mục lục

1. Mô tả ngắn
2. Yêu cầu
3. Cài đặt
4. Cấu trúc dự án (file quan trọng)
5. Cách chạy
6. Hướng dẫn sử dụng (người dùng / admin)
7. File dữ liệu & model
8. Chi tiết thuật toán phát hiện bất thường
9. Lưu ý vận hành & khắc phục lỗi
10. Người thực hiện / License

---

## 1. Mô tả ngắn

Ứng dụng được viết bằng Streamlit. Người dùng có thể:

* Chọn thông tin xe (hãng, dòng, loại, năm đăng ký, km, …) → bấm **Dự đoán giá** để nhận giá dự đoán.
* Nhập **Giá thực** để kiểm tra xem giá đó có bất thường hay không dựa trên `residual_z` so với thống kê phân khúc.
* Admin có thể mở màn hình **Danh sách xe giá bất thường**, xem, chọn, **Duyệt** hoặc **Xóa** từng tin hoặc tất cả cùng lúc; tải CSV.

---

## 2. Yêu cầu

* Python 3.8+
* Streamlit
* pandas, numpy, scikit-learn, joblib, matplotlib, seaborn
* openpyxl (nếu đọc file .xlsx)

Ví dụ `requirements.txt` tối thiểu:

```
streamlit>=1.18
pandas>=1.4
numpy>=1.21
scikit-learn>=1.0
joblib
matplotlib
seaborn
openpyxl
```

---

## 3. Cài đặt

```bash
# tạo virtualenv (khuyến nghị)
python -m venv .venv
source .venv/bin/activate   # mac/linux
.\.venv\Scripts\activate  # windows

# cài dependencies
pip install -r requirements.txt
```

---

## 4. Cấu trúc dự án (các file quan trọng)

```
.
├─ du_doan_gia_xe.py                          # file Streamlit chính
├─ data_motobikes.xlsx             # dataset mẫu (bắt buộc để load nếu không upload)
├─ data_motobikes_with_posts.csv   # (ghi ra khi publish)
├─ data_motobikes_updated.csv      # (ghi ra khi admin duyệt)
├─ motobike_price_model_project_1.pkl  # model dự đoán (joblib)
├─ residual_stats_by_group.csv     # bảng thống kê residual theo Dòng xe (mean,std,min,max,p10,p90)
├─ admin_queue.csv                 # hàng chờ admin (ghi/đọc)
├─ xe_may_cu.jpg                   # ảnh header
├─ requirements.txt
└─ README.md
```

> **Ghi chú:** đường dẫn file được xác định trong mã (biến `DATA_PATH`, `MODEL_PATH`, `STATS_PATH`, `ADMIN_PATH`, `UPDATED_PATH`). Nếu đổi tên file, cập nhật trong mã hoặc đặt đúng tên file trong cùng thư mục.

---

## 5. Cách chạy

Trong thư mục chứa `du_doan_gia_xe.py:

```bash
streamlit run du_doan_gia_xe.py
```

Mở trình duyệt theo địa chỉ do Streamlit cung cấp: https://dudoangiaxe-7k96wvwvrfsnjbi6gzujbe.streamlit.app/

---

## 6. Hướng dẫn sử dụng

### Trang **Giới thiệu**

* Mô tả dự án, mục tiêu, danh sách thành viên.

### Trang **Xây dựng mô hình**

* Mô tả pipeline tiền xử lý, các mô hình đã thử, kết quả (MAE, R²).
* Hiển thị bảng kết quả so sánh mô hình.

### Trang **Dự đoán giá xe**

1. Chọn thông tin xe (Thương hiệu, Dòng xe, Loại, Dung tích, Năm đăng ký, …).
2. Bấm **Dự đoán giá** → hệ thống sử dụng `motobike_price_model_project_1.pkl` để predict.

   * Nếu model trả về giá theo đơn vị triệu, mã nhân 1_000_000 để chuyển sang VND (heuristic kiểm tra median < 1e6).
3. Nhập **Giá muốn bán (VND)** → bấm **Kiểm tra bất thường**.

   * Nếu `std_ref == 0` hoặc NaN → cảnh báo không đủ dữ liệu tham chiếu.
   * Nếu `residual_z > 2` → **ĐẮT BẤT THƯỜNG**.
   * Nếu `residual_z < -2` → **RẺ BẤT THƯỜNG**.
   * Nếu `|residual_z| ≤ 2` → **BÌNH THƯỜNG**.

* Người dùng có thể **Đăng tin** (ghi vào `data_motobikes_with_posts.csv`) hoặc **Chuyển cho Admin** (ghi vào `admin_queue.csv`).

### Trang **Danh sách xe giá bất thường** (Admin)

* Ứng dụng:

  * Load dữ liệu, dự đoán giá toàn bộ dataset (vectorized nếu model hỗ trợ, fallback row-by-row).
  * Tính `Residual`, `Residual_z`, cờ vi phạm `min/max`, `p10/p90`, tính `_anomaly_score`.
  * Lọc tin bất thường theo điều kiện: `min/max` | `p10/p90` | `|Residual_z| >= 2` | `anomaly_score >= 60`.
  * Hiển thị theo trang (pagination), checkbox để chọn từng tin.
* Hành động Admin:

  * **Duyệt (chọn)**: ghi thông tin vào `data_motobikes_updated.csv` và cập nhật `admin_queue.csv` nếu tồn tại.
  * **Xóa (chọn)**: đánh dấu `Trạng_thái=deleted` trong `admin_queue.csv` (hoặc thêm record mới nếu chưa có).
  * **Duyệt TẤT CẢ / Xóa TẤT CẢ**: thao tác hàng loạt.
  * Sau duyệt/xóa, ứng dụng cập nhật `st.session_state['df_abnormal']` để UI chỉ hiện các tin còn lại (gọi `st.rerun()`).
  * Có nút **Tải CSV** danh sách bất thường.

---

## 7. File dữ liệu & model (chi tiết)

* `motobike_price_model_project_1.pkl` — model huấn luyện (joblib). Model phải nhận DataFrame với các cột `features`:

```
['Thương hiệu','Dòng xe','Tình trạng','Loại xe',
 'Dung tích xe','Năm đăng ký','Tuổi xe','Xuất xứ',
 'Chính sách bảo hành','Số Km đã đi']
```

* `residual_stats_by_group.csv` — bảng thống kê theo `Dòng xe` (index hoặc cột) nên chứa tối thiểu:

```
Dòng xe, mean, std, min, max, p10, p90
```

Nếu `Dòng xe` không có trong stats, ứng dụng dùng trung bình toàn bộ tập stats.

* `admin_queue.csv` — chứa hàng chờ admin: khi user chọn **Chuyển cho Admin**, app append một row:

```
Href, Thương hiệu, Dòng xe, Giá_thực_VND, Giá_dự_đoán_VND, Residual, Residual_z, Trạng_thái, Thời_gian
```

---

## 8. Chi tiết thuật toán phát hiện bất thường

* Residual = Giá_thực − Giá_dự_đoán
* residual_z = (residual − mean_ref) / std_ref

  * mean_ref, std_ref lấy từ stats theo Dòng xe (fallback toàn bộ)
* Ngưỡng:

  * residual_z > +2 → ĐẮT BẤT THƯỜNG
  * residual_z < −2 → RẺ BẤT THƯỜNG
  * |residual_z| ≤ 2 → BÌNH THƯỜNG

Trong trang admin, có thêm các kiểm tra: Min/max violation, p10/p90 violation, tính `_residual_score` (cap z ở 5), `_minmax_score`, `_p10p90_score` rồi gộp `_anomaly_score` với trọng số (w1=0.4, w2=0.4, w3=0.2). Lọc `anomaly_score >= 60`.

---

## 9. Lưu ý vận hành & khắc phục

* **Model load error**: nếu không load được model, app dừng và báo lỗi. Kiểm tra `MODEL_PATH` và định dạng pkl.
* **Excel/CSV**: sử dụng `openpyxl` để đọc `.xlsx`. Nếu đọc thất bại, thử xuất file sang CSV.
* **std_ref = 0 hoặc NaN**: app cảnh báo "Không đủ dữ liệu tham chiếu (std = 0)" và không đánh giá được residual_z.
* **Phiên bản model đổi cấu trúc features**: nếu model mới dùng tên cột khác, phải cập nhật phần `features` trong mã.
* **Session state**: app lưu `df_abnormal` & `admin_selected` trong `st.session_state` để duy trì trạng thái giữa tương tác. Nếu muốn reset, khởi động lại Streamlit hoặc xóa session state bằng code.
* **Ghi file CSV**: app thực hiện thao tác append / overwrite. Kiểm tra phân quyền ghi nếu chạy trên server.

---

## 10. Người thực hiện

* Mai Bảo Ngọc — Xây dựng mô hình dự báo giá
* Bùi Ngọc Toản — Xây dựng mô hình phát hiện bất thường
* Nguyễn Vũ Duy — Lập danh sách xe giá bất thường

---

## Gợi ý nâng cao (tùy chọn)

* Thêm endpoint API để gọi model từ service khác.
* Bảo mật: thêm xác thực cho trang admin.
* Lưu log hoạt động admin (ai duyệt/xóa, thời gian).
* Triển khai trên server (Heroku / Streamlit Cloud / VM) — nhớ cấu hình đường dẫn file, storage cho `admin_queue.csv` & `data_motobikes_updated.csv`.

---

### Tóm tắt nhanh (copy vào phần đầu README nếu muốn ngắn gọn)

```
# ỨNG DỤNG DỰ ĐOÁN GIÁ XE MÁY CŨ & PHÁT HIỆN GIÁ BẤT THƯỜNG

## Chạy
pip install -r requirements.txt
streamlit run app.py

## File quan trọng
- motobike_price_model_project_1.pkl
- residual_stats_by_group.csv
- data_motobikes.xlsx




