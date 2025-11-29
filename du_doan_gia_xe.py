import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import re
import os
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest
from difflib import SequenceMatcher
from math import ceil

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .main {
        padding-right: 0rem !important;
        padding-left: 0rem !important;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)




menu = ["Giới thiệu", "Xây dựng mô hình", "Dự đoán giá xe","Danh sách xe giá bất thường"]
choice = st.sidebar.selectbox('Menu', menu)    
if choice == 'Giới thiệu':
    st.markdown("### **ỨNG DỤNG DỰ ĐOÁN GIÁ XE MÁY CŨ VÀ PHÁT HIỆN GIÁ BẤT THƯỜNG**")
    st.image("xe_may_cu.jpg") 
    # --- PHẦN 1: DỰ ĐOÁN GIÁ XE ---
    st.markdown("### **DỰ ĐOÁN GIÁ XE**")
    st.markdown('<div class="bullet">• Ứng dụng cung cấp công cụ hỗ trợ định giá và gợi ý, giúp minh bạch hoá thị trường xe máy cũ và tăng tỉ lệ giao dịch thành công.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Hỗ trợ người bán định giá hợp lý cho xe máy cũ dựa trên các đặc điểm như thương hiệu, năm sản xuất, tình trạng và khu vực.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Giúp người mua so sánh và nhận diện mức giá hợp lý, tránh bị định giá quá cao.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Tối ưu hóa doanh thu và trải nghiệm người dùng cho Chợ Tốt thông qua việc gợi ý mức giá phù hợp, tăng khả năng giao dịch thành công.</div>', unsafe_allow_html=True)

    # --- KHOẢNG CÁCH GIỮA HAI PHẦN ---
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- PHẦN 2: DANH SÁCH XE GIÁ BẤT THƯỜNG ---
    st.markdown("### **DANH SÁCH XE GIÁ BẤT THƯỜNG**")
    st.markdown('<div class="bullet">• Giúp hệ thống nhanh chóng phát hiện những tin đăng có mức giá chênh lệch đáng kể so với mặt bằng chung của thị trường.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Hỗ trợ sàn giao dịch nhận diện các trường hợp định giá quá thấp (nguy cơ lừa đảo) hoặc quá cao (đặt giá sai lệch).</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Cho phép đội kiểm duyệt tập trung kiểm tra các tin đăng đáng nghi trước, tiết kiệm thời gian và nâng cao hiệu quả xử lý.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Góp phần đảm bảo tính minh bạch, giúp người mua yên tâm hơn khi lựa chọn xe và hạn chế các tin gây nhiễu trên sàn.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Bảo vệ người bán uy tín khỏi việc bị cạnh tranh không lành mạnh bởi các tin đăng đặt giá bất hợp lý.</div>', unsafe_allow_html=True)   
    # --- KHOẢNG CÁCH GIỮA HAI PHẦN ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # THÀNH VIÊN ---
    st.markdown("### **THÀNH VIÊN**")
    # dữ liệu
    data = {
        "STT": [1, 2, 3],
        "Họ tên": ["Mai Bảo Ngọc", "Bùi Ngọc Toản", "Nguyễn Vũ Duy"],
        "Vai trò": ["Xây dựng mô hình dự báo giá", "Xây dựng mô hình phát hiện bất thường", "Lập danh sách xe giá bất thường"]
    }
    df = pd.DataFrame(data)

    # hiển thị
    st.table(df.set_index("STT"))  
    
elif choice == 'Xây dựng mô hình':
    st.markdown("### **1. Tiền xử lý dữ liệu**")

    st.markdown("""
    Bộ dữ liệu xe máy cũ được thu thập từ nền tảng *Chợ Tốt*, bao gồm các thuộc tính phản ánh đặc điểm kỹ thuật, mức độ sử dụng và giá rao bán của xe. 
    Trước khi đưa vào mô hình dự báo, dữ liệu được xử lý và chuẩn hóa theo quy trình sau:
    """)

    st.markdown("""
    <ul style="line-height: 1.8;">
    <li>Chuẩn hóa các trường giá (<i>Giá</i>, <i>Khoảng giá min</i>, <i>Khoảng giá max</i>) nhằm đảm bảo tính nhất quán khi phân tích.</li>
    <li>Loại bỏ các bản ghi thiếu dữ liệu quan trọng hoặc chứa giá trị ngoại lai gây ảnh hưởng đến chất lượng mô hình.</li>
    <li>Chuẩn hóa kiểu dữ liệu cho các biến số như <i>Năm đăng ký</i>, <i>Số Km đã đi</i>, … để đảm bảo tương thích với các thuật toán học máy.</li>
    <li>Thực hiện scaling cho các biến liên tục nhằm giảm sai lệch thang đo và cải thiện độ ổn định trong quá trình huấn luyện.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown(""" 
    Các biến phân loại (<i>Thương hiệu</i>, <i>Dòng xe</i>, <i>...</i>) được xử lý bằng <b>StringIndexer</b> và <b>OneHotEncoder</b>. 
    Sau đó toàn bộ đặc trưng được hợp nhất thành một vector đầu vào duy nhất thông qua <b>VectorAssembler</b>.
    """, unsafe_allow_html=True)

    st.markdown("""
    Dữ liệu sau khi chuẩn hóa được chia theo tỷ lệ:
    """)

    st.markdown("""
    <ul style="line-height: 1.8;">
    <li><b>80%</b> dùng để huấn luyện mô hình.</li>
    <li><b>20%</b> dùng để đánh giá hiệu suất dự báo.</li>
    </ul>
    """, unsafe_allow_html=True)




# --- XÂY DỰNG MÔ HÌNH DỰ BÁO GIÁ ---

    st.markdown("### **2. Xây dựng mô hình dự báo giá**")

    st.markdown("""
    Nhóm tiến hành huấn luyện nhiều thuật toán khác nhau nhằm so sánh hiệu năng và lựa chọn mô hình tối ưu, bao gồm:
    """)

    # Bullet list các thuật toán
    st.markdown("""
    <ul style="line-height: 1.8;">
    <li>Linear Regression</li>
    <li>Random Forest Regressor</li>
    <li>Gradient Boosted Trees (GBT)</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    Tất cả các mô hình đều được đánh giá bằng cùng một bộ thước đo:
    """)

    # Bullet list các chỉ số đánh giá
    st.markdown("""
    <ul style="line-height: 1.8;">
    <li><b>MAE (Mean Absolute Error)</b>: sai số dự báo trung bình tuyệt đối giữa giá trị thực tế và giá trị dự đoán.</li>
    <li><b>R² (hệ số xác định)</b>: độ phù hợp của mô hình (càng cao càng tốt).</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("Kết quả huấn luyện mô hình:")

    # --- BẢNG KẾT QUẢ ---
    import pandas as pd

    results = {
        "Mô hình": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosted Trees (GBT)"
        ],
        "MAE (VND)": [
            "6.700.876.000",
            "5.744.014",
            "7.142.370",
        ],
        "R²": [
            "-5,98e+19",
            "0,7518",
            "0,6962",
        ],
        "Nhận xét": [
            "Sai số cực lớn và R² âm nên mô hình hoàn toàn không phù hợp với dữ liệu",
            "Sai số thấp nhất và R² cao nhất, là mô hình cho hiệu suất tốt nhất",
            "Sai số và R² ở mức khá, nhưng vẫn kém hơn Random Forest",
        ]
    }

    df_result = pd.DataFrame(results)

    # Ẩn index
    st.dataframe(df_result, hide_index=True)
    
    st.markdown("""
    Kết quả so sánh mô hình cho thấy Random Forest hoạt động tốt nhất trong ba mô hình, với giá trị MAE ≈ 5.74 và R² ≈ 0.75, cho thấy mô hình giải thích được khoảng 75% phương sai của dữ liệu giá xe và có sai số dự đoán trung bình thấp nhất. Mô hình Gradient Boosting đứng thứ hai, có độ chính xác khá tốt nhưng kém hơn một chút so với Random Forest (MAE ≈ 7.14, R² ≈ 0.70). Ngược lại, Linear Regression cho kết quả rất kém, với MAE cực lớn, R² âm (≈ –5.98e+19), chứng tỏ mô hình tuyến tính không phù hợp với tập dữ liệu này – có thể do mối quan hệ giữa các biến độc lập và giá xe là phi tuyến tính và phức tạp. Như vậy, Random Forest là lựa chọn tối ưu để dự đoán giá xe máy trong trường hợp này.
    """)
    
    st.markdown("### **3. Phát hiện xe máy giá bất thường**")

    st.markdown("""
    Quy trình kiểm tra một mức giá có bất thường hay không được thực hiện dựa trên mô hình dự đoán và thống kê theo từng dòng xe. 
    Hệ thống vận hành theo các bước sau:

    #### **Bước 1 — Nhập giá thực tế từ người dùng**
    Người dùng cung cấp mức giá rao bán để hệ thống so sánh với giá dự đoán và dữ liệu tham chiếu.

    #### **Bước 2 — Kiểm tra điều kiện trước khi đánh giá**
    Hệ thống yêu cầu phải có giá dự đoán của xe (từ mô hình dự báo) trước khi tiến hành kiểm tra.

    #### **Bước 3 — Tính sai số dự báo (Residual)**
    Sai số được tính bằng chênh lệch giữa giá thực và giá dự đoán:
    
    **residual = Giá_thực − Giá_dự_đoán**

    #### **Bước 4 — Lấy giá trị tham chiếu theo dòng xe**
    Hệ thống sử dụng bảng thống kê residual theo từng dòng xe để lấy:
    - mean residual (mean_ref)
    - độ lệch chuẩn residual (std_ref)

    Nếu dòng xe không có dữ liệu, hệ thống dùng trung bình toàn bộ tập dữ liệu.

    #### **Bước 5 — Chuẩn hoá sai số (Residual-z)**
    Sai số được chuẩn hoá để đánh giá mức độ lệch so với thị trường của phân khúc:

    **residual_z = (residual − mean_ref) / std_ref**

    Giá trị này giúp xác định mức giá có lệch bất thường so với nhóm xe tương đồng hay không.

    #### **Bước 6 — Đánh giá bất thường**
    Dựa trên ngưỡng chuẩn hoá:
    - **residual_z > +2** → Giá **đắt bất thường**
    - **residual_z < −2** → Giá **rẻ bất thường**
    - **|residual_z| ≤ 2** → Giá **bình thường**

    Kết quả giúp người dùng và hệ thống nhận diện các tin đăng rao bán lệch so với mặt bằng chung của thị trường.
    """, unsafe_allow_html=True)


    st.markdown("## **4. Lập danh sách tổng hợp các xe có giá bất thường**")

    st.markdown("""
    Bên cạnh việc kiểm tra giá cho từng xe theo yêu cầu của người dùng, hệ thống còn cung cấp chức năng **liệt kê toàn bộ các tin đăng có mức giá bất thường** nhằm hỗ trợ công tác kiểm duyệt của quản trị viên. 
    Mục tiêu của tính năng này là giúp admin nhanh chóng phát hiện những tin rao bán lệch khỏi mặt bằng thị trường và đảm bảo chất lượng dữ liệu trên sàn giao dịch.
    """)

    st.markdown("### **Thông tin hiển thị trong danh sách**")
    st.markdown("""
    Mỗi xe trong danh sách bất thường được trình bày kèm theo:
    <ul style="line-height:1.7;">
    <li><b>Giá thực tế</b> và <b>giá dự đoán</b> từ mô hình.</li>
    <li><b>Residual</b> (mức chênh lệch tuyệt đối).</li>
    <li><b>Residual-z</b>, thể hiện mức độ bất thường theo đơn vị độ lệch chuẩn.</li>
    <li>Thông tin mô tả xe: thương hiệu, dòng xe, loại xe,… để admin đối chiếu nhanh.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("### **Quy trình xử lý của admin**")
    st.markdown("""
    - Admin có thể xem chi tiết từng tin đăng, kiểm tra mô tả và hình ảnh, sau đó đưa ra quyết định: phê duyệt, xác minh lại hoặc từ chối.  
    - Danh sách cung cấp nút tải xuống CSV để phục vụ công tác kiểm tra hàng loạt và lưu trữ hồ sơ kiểm duyệt.
    """)

    st.markdown("### **Lợi ích**")
    st.markdown("""
    <ul style="line-height:1.7;">
    <li>Ngăn chặn các tin rao có giá quá thấp hoặc quá cao một cách bất hợp lý, giảm nhiễu thị trường.</li>
    <li>Hỗ trợ phát hiện sớm các tin có dấu hiệu gian lận hoặc thiếu minh bạch.</li>
    <li>Bảo vệ người mua bằng cách cảnh báo các mức giá không phù hợp.</li>
    <li>Giúp đội kiểm duyệt làm việc hiệu quả hơn, duy trì chất lượng và tính minh bạch của sàn giao dịch.</li>
    </ul>
    """, unsafe_allow_html=True)
    





elif choice == 'Dự đoán giá xe':
    ###### Giao diện Streamlit ######
    st.image("xe_may_cu.jpg", use_container_width=True)
    st.title("Dự đoán giá xe máy")


    # load model dự đoán giá
    @st.cache_resource(ttl=3600)
    def load_model(path="bmotobike_price_model_project_1.pkl"):
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            st.error(f"Không thể load model từ {path}: {e}")
            return None

    model = load_model("motobike_price_model_project_1.pkl")  


    # đọc dữ liệu từ file data_motobikes.xlsx
    df = pd.read_excel("data_motobikes.xlsx")
    st.dataframe(df.head())   

    # Trường hợp 2: Đọc dữ liệu từ file csv, excel do người dùng tải lên
    st.write("### Tải file dữ liệu")

    uploaded_file = st.file_uploader(
        "Chọn file dữ liệu (CSV hoặc Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Dữ liệu đã nhập:")
        st.dataframe(df.head())
        
        
    st.write("### 1. Dự đoán giá xe máy cũ")
    

    # Chọn hãng xe
    thuong_hieu = st.selectbox(
        "Chọn hãng xe",
        df['Thương hiệu'].unique()
    )

    # Lọc dữ liệu theo hãng vừa chọn
    df_filtered = df[df['Thương hiệu'] == thuong_hieu]
    # Chọn dòng xe phụ thuộc vào hãng
    dong_xe = st.selectbox(
        "Chọn dòng xe",
        df_filtered['Dòng xe'].unique()    
    )
    tinh_trang = st.selectbox("Chọn tình trạng", df['Tình trạng'].unique())

    # Lọc dữ liệu theo dòng xe vừa chọn
    df_filtered_by_dong = df_filtered[df_filtered['Dòng xe'] == dong_xe]
    # Chọn loại xe phụ thuộc vào dòng xe
    loai_xe = st.selectbox(
        "Chọn loại xe",
        df_filtered_by_dong['Loại xe'].unique()
    )
    dung_tich_xi_lanh = st.selectbox("Dung tích xi lanh (cc)", df['Dung tích xe'].unique())
    nam_dang_ky = st.slider("Năm đăng ký", 2000, 2024, 2015)
    Tuoi_xe = datetime.now().year - nam_dang_ky
    xuat_xu = st.selectbox("Xuất xứ", df['Xuất xứ'].unique())
    chinh_sach_bao_hanh = st.selectbox("Chính sách bảo hành", df['Chính sách bảo hành'].unique())
    so_Km_da_di = st.number_input("Số Km đã đi", min_value=0, max_value=200000, value=50000, step=1000)
    du_doan_gia = st.button("Dự đoán giá")




    if du_doan_gia:
        input_data = pd.DataFrame([{
            'Thương hiệu': thuong_hieu,
            'Dòng xe': dong_xe,
            'Tình trạng': tinh_trang,
            'Loại xe': loai_xe,
            'Dung tích xe': dung_tich_xi_lanh,
            'Năm đăng ký': nam_dang_ky,
            'Tuổi xe': Tuoi_xe,
            'Xuất xứ': xuat_xu,
            'Chính sách bảo hành': chinh_sach_bao_hanh,
            'Số Km đã đi': so_Km_da_di
        }])

        # Dự đoán bằng model đã load
        y_pred = model.predict(input_data)

        gia_du_doan = float(y_pred[0])

        # Nếu mô hình của bạn dự đoán theo triệu → đổi ra VND
        gia_du_doan_vnd = int(gia_du_doan * 1_000_000)

        st.success(f"Giá dự đoán: {gia_du_doan_vnd:,.0f} VND")
        st.session_state['gia_du_doan_vnd'] = gia_du_doan_vnd





    st.write("### 2. Phát hiện xe máy giá bất thường")
    stats = pd.read_csv("residual_stats_by_group.csv", index_col=0)



    gia_thuc = st.number_input(
        "Nhập giá muốn bán (VND):",
        min_value=0,
        value=15_000_000,
        step=100_000,
        key="gia_thuc_input"
    )

    # nút để người dùng chủ động yêu cầu kiểm tra
    kiem_tra = st.button("Kiểm tra bất thường")

    # chỉ khi bấm nút mới tính và hiển thị kết quả
    if kiem_tra:
        if "gia_du_doan_vnd" not in st.session_state:
            st.info("Hãy bấm 'Dự đoán giá' trước để có giá dự đoán.")
        else:
            gia_du_doan_vnd = st.session_state["gia_du_doan_vnd"]
            # đảm bảo có dong_xe nếu cần dùng để tra stats
            dong_xe = st.session_state.get("dong_xe", dong_xe if 'dong_xe' in locals() else None)

            residual = gia_thuc - gia_du_doan_vnd

            # lấy mean/std từ stats (đã load từ CSV trước đó)
            if dong_xe is not None and dong_xe in stats.index:
                mean_ref = stats.loc[dong_xe, "mean"]
                std_ref  = stats.loc[dong_xe, "std"]
            else:
                mean_ref = stats["mean"].mean()
                std_ref  = stats["std"].mean()

            if pd.isna(std_ref) or std_ref == 0:
                st.warning("Không đủ dữ liệu tham chiếu (std = 0). Không thể đánh giá bất thường.")
            else:
                residual_z = (residual - mean_ref) / std_ref

                # --- UI hiển thị kết quả (giống mockup) ---
                if residual_z > 2:
                    st.error(f"⚠️ PHÁT HIỆN BẤT THƯỜNG: Giá CAO bất thường.")
                    st.info("Tin này có mức giá chênh lệch lớn so với thị trường. Tin sẽ **KHÔNG** được đăng ngay mà phải chuyển qua Admin duyệt.")
                    # hai lựa chọn: nhập lại hoặc chuyển admin
                    col1, col2 = st.columns([1,1])
                    with col1:
                        if st.button("✏️ Nhập lại"):
                            # xóa giá đã nhập, focus cho người dùng nhập lại
                            # set lại input về giá mặc định hoặc None
                            st.session_state["gia_thuc_input"] = 0
                            st.rerun()
                    with col2:
                        if st.button("⚠️ Xác nhận: Chuyển cho Admin"):
                            # chuẩn bị bản ghi để gửi admin
                            try:
                                admin_row = {
                                    "Thương hiệu": thuong_hieu if 'thuong_hieu' in locals() else None,
                                    "Dòng xe": dong_xe,
                                    "Loại xe": loai_xe if 'loai_xe' in locals() else None,
                                    "Giá_thực_VND": gia_thuc,
                                    "Giá_dự_đoán_VND": gia_du_doan_vnd,
                                    "Residual": residual,
                                    "Residual_z": residual_z,
                                    "Trạng_thái": "pending_review",
                                    "Thời_gian": datetime.now().isoformat()
                                }
                                # lưu vào file hàng chờ admin (append CSV)
                                admin_path = "admin_queue.csv"
                                if os.path.exists(admin_path):
                                    pd.concat([pd.read_csv(admin_path), pd.DataFrame([admin_row])], ignore_index=True).to_csv(admin_path, index=False, encoding="utf-8")
                                else:
                                    pd.DataFrame([admin_row]).to_csv(admin_path, index=False, encoding="utf-8")
                                st.success("Đã chuyển tin tới Admin để phê duyệt.")
                            except Exception as e:
                                st.error(f"Lỗi khi chuyển Admin: {e}")

                elif residual_z < -2:
                    st.error(f"⚠️ PHÁT HIỆN BẤT THƯỜNG: Giá RẺ bất thường.")
                    st.info("Tin có giá thấp bất thường. Tin sẽ **KHÔNG** được đăng ngay mà phải chuyển qua Admin duyệt.")
                    col1, col2 = st.columns([1,1])
                    with col1:
                        if st.button("✏️ Nhập lại"):
                            st.session_state["gia_thuc_input"] = 0
                            st.rerun()
                    with col2:
                        if st.button("⚠️ Xác nhận: Chuyển cho Admin"):
                            try:
                                admin_row = {
                                    "Thương hiệu": thuong_hieu if 'thuong_hieu' in locals() else None,
                                    "Dòng xe": dong_xe,
                                    "Loại xe": loai_xe if 'loai_xe' in locals() else None,
                                    "Giá_thực_VND": gia_thuc,
                                    "Giá_dự_đoán_VND": gia_du_doan_vnd,
                                    "Residual": residual,
                                    "Residual_z": residual_z,
                                    "Trạng_thái": "pending_review",
                                    "Thời_gian": datetime.now().isoformat()
                                }
                                admin_path = "admin_queue.csv"
                                if os.path.exists(admin_path):
                                    pd.concat([pd.read_csv(admin_path), pd.DataFrame([admin_row])], ignore_index=True).to_csv(admin_path, index=False, encoding="utf-8")
                                else:
                                    pd.DataFrame([admin_row]).to_csv(admin_path, index=False, encoding="utf-8")
                                st.success("Đã chuyển tin tới Admin để phê duyệt.")
                            except Exception as e:
                                st.error(f"Lỗi khi chuyển Admin: {e}")

                else:
                    st.success("✔ Bình thường")
                    # nút đăng tin (append vào dataset chính)
                    if st.button("✅ Đăng tin"):
                        try:
                            # tạo bản ghi để thêm (bạn có thể bổ sung các trường khác từ form)
                            new_row = {
                                "Thương hiệu": thuong_hieu if 'thuong_hieu' in locals() else None,
                                "Dòng xe": dong_xe,
                                "Loại xe": loai_xe if 'loại_xe' in locals() else None,
                                "Năm đăng ký": nam_dang_ky if 'nam_dang_ky' in locals() else None,
                                "Tuổi xe": Tuoi_xe if 'Tuoi_xe' in locals() else None,
                                "Số Km đã đi": so_Km_da_di if 'so_Km_da_di' in locals() else None,
                                "Giá": gia_thuc,
                                "Giá_dự_doán": gia_du_doan_vnd,
                                "Residual": residual,
                                "Residual_z": residual_z,
                                "Trạng_thái": "published",
                                "Thời_gian": datetime.now().isoformat()
                            }
                            # append vào df (in-memory)
                            try:
                                df = df.append(new_row, ignore_index=True)
                            except Exception:
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            # lưu file backup / updated
                            save_path = "data_motobikes_with_posts.csv"
                            df.to_csv(save_path, index=False, encoding="utf-8")
                            st.success(f"Đã đăng tin và thêm vào dataset ({save_path}).")
                        except Exception as e:
                            st.error(f"Lỗi khi đăng tin: {e}")

                    
                    
                                  
elif choice == 'Danh sách xe giá bất thường':
    st.write("### Danh sách các xe bất thường trong tập dữ liệu")
    st.caption("Admin xem xét các tin giá lệch cao/thấp để phê duyệt hoặc xóa.")

    # Imports local to this block (an app có thể đã import ở đầu file; đưa vào để đoạn này standalone)
    import uuid
    from math import ceil
    from datetime import datetime

    # ---------- FILE PATHS ----------
    DATA_PATH = "data_motobikes.xlsx"
    MODEL_PATH = "motobike_price_model_project_1.pkl"
    STATS_PATH = "residual_stats_by_group.csv"
    ADMIN_PATH = "admin_queue.csv"
    UPDATED_PATH = "data_motobikes_updated.csv"

    # ---------- 0. LOAD / SAFETY ----------
    # df (cố gắng load nếu chưa có)
    if 'df' not in globals() and 'df' not in locals():
        try:
            df = pd.read_excel(DATA_PATH)
            st.info("Đã load dữ liệu từ data_motobikes.xlsx")
        except Exception as e:
            st.error(f"Không tìm thấy DataFrame 'df' và không thể load file data_motobikes.xlsx: {e}")
            st.stop()

    # model (cố gắng load nếu chưa có)
    if 'model' not in globals() and 'model' not in locals():
        try:
            model = joblib.load(MODEL_PATH)
            st.info("Đã load model dự đoán.")
        except Exception as e:
            st.error(f"Không tìm thấy model và không thể load {MODEL_PATH}: {e}")
            st.stop()

    # stats
    if 'stats' not in globals() and 'stats' not in locals():
        try:
            stats = pd.read_csv(STATS_PATH, index_col=0)
            st.info("Đã load residual_stats_by_group.csv")
        except Exception as e:
            st.error(f"Không tìm thấy {STATS_PATH}: {e}")
            st.stop()

    # admin queue (nếu không tồn tại -> tạo DataFrame rỗng)
    if os.path.exists(ADMIN_PATH):
        try:
            admin_q = pd.read_csv(ADMIN_PATH, dtype=str)
        except Exception:
            admin_q = pd.DataFrame()
    else:
        admin_q = pd.DataFrame()

    # ---------- 1. PREPROCESS COPY (không sửa df gốc) ----------
    df_local = df.copy()
    # chuẩn hóa Giá -> numeric (loại bỏ ký tự không phải số)
    if 'Giá' in df_local.columns:
        df_local["Giá"] = df_local["Giá"].astype(str).str.replace(r"[^\d]", "", regex=True)
        df_local["Giá"] = pd.to_numeric(df_local["Giá"], errors="coerce")
    else:
        df_local["Giá"] = np.nan

    # Năm đăng ký -> numeric, Tuổi xe
    if 'Năm đăng ký' in df_local.columns:
        df_local["Năm đăng ký"] = pd.to_numeric(df_local["Năm đăng ký"], errors="coerce")
        df_local["Tuổi xe"] = datetime.now().year - df_local["Năm đăng ký"]
    else:
        df_local["Tuổi xe"] = np.nan

    # ensure Href column exists (khóa nhận dạng)
    if 'Href' not in df_local.columns:
        df_local['Href'] = [str(uuid.uuid4()) for _ in range(len(df_local))]

    # ---------- 2. DỰ ĐOÁN: vectorized nếu model hỗ trợ, fallback hàng loạt ----------
    features = [
        'Thương hiệu','Dòng xe','Tình trạng','Loại xe',
        'Dung tích xe','Năm đăng ký','Tuổi xe','Xuất xứ',
        'Chính sách bảo hành','Số Km đã đi'
    ]

    with st.spinner("Đang dự đoán cho toàn bộ dataset ..."):
        try:
            X = df_local[features]
            y_hat = model.predict(X)  # nhiều model trả về triệu → nhân nếu cần
            try:
                arr = np.array(y_hat, dtype=float)
                if np.nanmedian(arr) < 1e6:
                    arr = arr * 1_000_000
                df_local["Giá dự đoán"] = arr
            except Exception:
                df_local["Giá dự đoán"] = y_hat
        except Exception:
            # fallback row-by-row
            preds = []
            for _, r in df_local.iterrows():
                x = pd.DataFrame([r[features].to_dict()])
                try:
                    y = model.predict(x)[0]
                    y = float(y)
                    if y < 1e6:  # heuristic: nhân triệu nếu trả về triệu
                        y = y * 1_000_000
                    preds.append(y)
                except Exception:
                    preds.append(np.nan)
            df_local["Giá dự đoán"] = preds

    # ---------- 3. TÍNH RESIDUAL + JOIN STATS THEO 'Dòng xe' ----------
    df_local["Residual"] = df_local["Giá"] - df_local["Giá dự đoán"]

    # stats có thể đã có index = Dòng xe hoặc là bảng có cột 'Dòng xe'
    if "Dòng xe" in stats.columns:
        stats_idx = stats.set_index("Dòng xe")
    else:
        stats_idx = stats

    # join (left) theo 'Dòng xe'
    df_local = df_local.join(stats_idx, on="Dòng xe", how="left")

    # tính Residual_z, cẩn trọng std = 0 / NaN
    df_local["Residual_z"] = (df_local["Residual"] - df_local.get("mean", np.nan)) / df_local.get("std", np.nan)
    df_local["Residual_z"] = df_local["Residual_z"].replace([np.inf, -np.inf], np.nan)

    # ---------- 4. CÁC CỜ VI PHẠM & ANOMALY SCORE ----------
    df_local["_minmax_violation"] = 0
    df_local["_p10p90_violation"] = 0

    has_min = "min" in df_local.columns
    has_max = "max" in df_local.columns
    has_p10 = "p10" in df_local.columns
    has_p90 = "p90" in df_local.columns

    cond_min = pd.Series(False, index=df_local.index)
    cond_max = pd.Series(False, index=df_local.index)
    if has_min:
        cond_min = pd.notna(df_local["min"]) & (df_local["Giá"] < df_local["min"])
    if has_max:
        cond_max = pd.notna(df_local["max"]) & (df_local["Giá"] > df_local["max"])
    df_local.loc[cond_min | cond_max, "_minmax_violation"] = 1

    cond_p10 = pd.Series(False, index=df_local.index)
    cond_p90 = pd.Series(False, index=df_local.index)
    if has_p10:
        cond_p10 = pd.notna(df_local["p10"]) & (df_local["Giá"] < df_local["p10"])
    if has_p90:
        cond_p90 = pd.notna(df_local["p90"]) & (df_local["Giá"] > df_local["p90"])
    df_local.loc[cond_p10 | cond_p90, "_p10p90_violation"] = 1

    # residual score (cap z to 5)
    cap_z = 5.0
    df_local["_residual_score"] = df_local["Residual_z"].abs().fillna(0).clip(upper=cap_z) / cap_z * 100
    df_local["_minmax_score"] = df_local["_minmax_violation"] * 100
    df_local["_p10p90_score"] = df_local["_p10p90_violation"] * 100

    w1, w2, w3 = 0.40, 0.40, 0.20
    df_local["_anomaly_score"] = (
        w1 * df_local["_residual_score"] +
        w2 * df_local["_minmax_score"] +
        w3 * df_local["_p10p90_score"]
    )
    df_local["_anomaly_score"] = df_local["_anomaly_score"].clip(0, 100)

    # ---------- 5. LỌC BẤT THƯỜNG (theo điều kiện bạn định nghĩa) ----------
    cond_minmax = df_local["_minmax_violation"] == 1
    cond_percentile = df_local["_p10p90_violation"] == 1
    cond_residualz = df_local["Residual_z"].abs() >= 2
    cond_score = df_local["_anomaly_score"] >= 60

    df_abnormal_new = df_local[cond_minmax | cond_percentile | cond_residualz | cond_score].copy()
    df_abnormal_new = df_abnormal_new.reset_index(drop=True)

    # lưu vào session_state để duy trì trạng thái UI giữa các lần thao tác admin
    if 'df_abnormal' not in st.session_state:
        st.session_state['df_abnormal'] = df_abnormal_new.copy()
    else:
        # nếu session đã có, ta dùng phiên bản hiện tại (đảm bảo không override các thay đổi admin trước đó)
        # nhưng vẫn cập nhật nếu có bản mới (ví dụ thêm tin mới từ nguồn) — ở đây ta giữ phiên hiện tại
        pass

    # dùng bản trong session_state cho phần hiển thị
    df_abnormal = st.session_state.get('df_abnormal', df_abnormal_new).reset_index(drop=True)

    # thêm cột Nhận định
    def decide_label(row):
        if row.get("_minmax_violation", 0) == 1:
            if pd.notna(row.get("min")) and row["Giá"] < row["min"]:
                return "RẺ BẤT THƯỜNG"
            if pd.notna(row.get("max")) and row["Giá"] > row["max"]:
                return "ĐẮT BẤT THƯỜNG"
        if row.get("_p10p90_violation", 0) == 1:
            if pd.notna(row.get("p10")) and row["Giá"] < row["p10"]:
                return "RẺ BẤT THƯỜNG"
            if pd.notna(row.get("p90")) and row["Giá"] > row["p90"]:
                return "ĐẮT BẤT THƯỜNG"
        if pd.notna(row.get("Residual")):
            return "ĐẮT BẤT THƯỜNG" if row["Residual"] > 0 else "RẺ BẤT THƯỜNG"
        return "BÌNH THƯỜNG"

    if not df_abnormal.empty:
        df_abnormal["Nhận định"] = df_abnormal.apply(decide_label, axis=1)

    # ---------- 6. HIỂN THỊ (và admin controls) ----------
    total_abn = len(df_abnormal)
    if total_abn == 0:
        st.success("✔ Không có xe bất thường trong dataset.")
        # show pending admin queue briefly
        if not admin_q.empty:
            st.markdown("#### Hàng chờ admin")
            st.dataframe(admin_q.head(20))
        st.stop()

    # show header + pending count
    pending_count = 0
    if not admin_q.empty and 'Trạng_thái' in admin_q.columns:
        pending_count = int((admin_q['Trạng_thái'] == 'pending_review').sum())
    if pending_count > 0:
        st.warning(f"Cảnh báo: Có {pending_count} tin cần duyệt.")

    st.markdown(f"### Danh sách xe giá bất thường ({total_abn} tin tổng)")

    # --- Pagination ---
    PER_PAGE = 10
    total = total_abn
    total_pages = max(1, ceil(total / PER_PAGE))
    if 'abn_page' not in st.session_state:
        st.session_state['abn_page'] = 0
    page = st.session_state['abn_page']
    page = min(page, total_pages - 1)
    start = page * PER_PAGE
    end = start + PER_PAGE
    df_page = df_abnormal.iloc[start:end].reset_index(drop=True)

    # selection state
    if 'admin_selected' not in st.session_state:
        st.session_state['admin_selected'] = {}
    # ensure keys for current page
    for href in df_page['Href'].astype(str).tolist():
        if href not in st.session_state['admin_selected']:
            st.session_state['admin_selected'][href] = False

    st.write("Chọn các tin để thực hiện hành động:")
    h1, h2, h3, h4, h5 = st.columns([0.05, 0.60, 0.12, 0.12, 0.11])
    h1.markdown("**Chọn**")
    h2.markdown("**Tiêu đề**")
    h3.markdown("**Giá thực**")
    h4.markdown("**Giá dự đoán**")
    h5.markdown("**Residual_z**")

    # render rows with checkbox
    for idx, row in df_page.iterrows():
        cols = st.columns([0.05, 0.60, 0.12, 0.12, 0.11])  # adjust widths
        href = str(row['Href'])
        checked = cols[0].checkbox("", value=st.session_state['admin_selected'].get(href, False), key=f"chk_{href}")
        st.session_state['admin_selected'][href] = checked

        # title + meta
        with cols[1]:
            title = row.get('Tiêu đề') if 'Tiêu đề' in row.index else ""
            if pd.isna(title) or title == "":
                title = f"{row.get('Thương hiệu','')} {row.get('Dòng xe','')}"
            st.markdown(f"**{title}**")
            meta = []
            if 'Thương hiệu' in row and pd.notna(row.get('Thương hiệu')): meta.append(str(row.get('Thương hiệu')))
            if 'Dòng xe' in row and pd.notna(row.get('Dòng xe')): meta.append(str(row.get('Dòng xe')))
            if meta:
                st.caption(" / ".join(meta))

        # Giá
        with cols[2]:
            try:
                g = int(row.get('Giá')) if not pd.isna(row.get('Giá')) else None
                st.write(f"{g:,.0f} ₫" if g is not None else "N/A")
            except Exception:
                st.write(row.get('Giá'))

        # Giá dự đoán
        with cols[3]:
            try:
                gd = int(row.get('Giá dự đoán')) if not pd.isna(row.get('Giá dự đoán')) else None
                st.write(f"{gd:,.0f} ₫" if gd is not None else "N/A")
            except Exception:
                st.write(row.get('Giá dự đoán'))

        # Residual_z
        with cols[4]:
            rz = row.get('Residual_z')
            if pd.isna(rz):
                st.write("N/A")
            else:
                st.write(f"{rz:.2f}")

    st.markdown("---")

    # --- Admin action buttons ---
    colA, colB, colC, colD = st.columns(4)

    def get_selected_hrefs():
        return [href for href, v in st.session_state['admin_selected'].items() if v]

    # Approve selected
    with colA:
        if st.button("✅ Duyệt (chọn)"):
            selected = get_selected_hrefs()
            if not selected:
                st.info("Chưa chọn tin nào để duyệt.")
            else:
                # load or create updated file
                if os.path.exists(UPDATED_PATH):
                    try:
                        df_upd = pd.read_csv(UPDATED_PATH, dtype=str)
                    except Exception:
                        df_upd = pd.DataFrame()
                else:
                    df_upd = pd.DataFrame()

                cnt = 0
                for href in selected:
                    rows = df_abnormal[df_abnormal['Href'].astype(str) == href]
                    if rows.empty:
                        continue
                    rec = rows.iloc[0].copy()
                    rec['Trạng_thái'] = 'approved'
                    rec['Thời_gian'] = datetime.now().isoformat()
                    rec_df = pd.DataFrame([rec])
                    if ('Href' in df_upd.columns) and ((df_upd['Href'].astype(str) == href).any()):
                        df_upd.loc[df_upd['Href'].astype(str) == href, rec_df.columns] = rec_df.iloc[0].values
                    else:
                        df_upd = pd.concat([df_upd, rec_df], ignore_index=True, sort=False)
                    # update admin queue if exists
                    if os.path.exists(ADMIN_PATH):
                        try:
                            aq = pd.read_csv(ADMIN_PATH, dtype=str)
                            if 'Href' in aq.columns and (aq['Href'].astype(str) == href).any():
                                aq.loc[aq['Href'].astype(str) == href, 'Trạng_thái'] = 'approved'
                                aq.to_csv(ADMIN_PATH, index=False, encoding='utf-8')
                        except Exception:
                            pass
                    cnt += 1
                try:
                    df_upd.to_csv(UPDATED_PATH, index=False, encoding='utf-8')
                except Exception as e:
                    st.error(f"Lỗi khi lưu file cập nhật: {e}")

                # remove approved from df_abnormal in session_state so UI cập nhật
                cur_df = st.session_state.get('df_abnormal', pd.DataFrame()).copy()
                st.session_state['df_abnormal'] = cur_df.loc[~cur_df['Href'].astype(str).isin(selected)].reset_index(drop=True)

                st.success(f"Đã duyệt {cnt} tin; ghi nhận vào {UPDATED_PATH}.")
                for href in selected:
                    st.session_state['admin_selected'][href] = False
                st.rerun()

    # Delete selected
    with colB:
        if st.button("🗑️ Xóa (chọn)"):
            selected = get_selected_hrefs()
            if not selected:
                st.info("Chưa chọn tin nào để xóa.")
            else:
                if os.path.exists(ADMIN_PATH):
                    try:
                        aq = pd.read_csv(ADMIN_PATH, dtype=str)
                    except Exception:
                        aq = pd.DataFrame()
                else:
                    aq = pd.DataFrame()
                cnt = 0
                for href in selected:
                    rows = df_abnormal[df_abnormal['Href'].astype(str) == href]
                    if rows.empty:
                        continue
                    rec = rows.iloc[0]
                    if not aq.empty and 'Href' in aq.columns and (aq['Href'].astype(str) == href).any():
                        aq.loc[aq['Href'].astype(str) == href, 'Trạng_thái'] = 'deleted'
                    else:
                        newrow = {
                            "Href": href,
                            "Thương hiệu": rec.get('Thương hiệu'),
                            "Dòng xe": rec.get('Dòng xe'),
                            "Giá_thực_VND": rec.get('Giá'),
                            "Giá_dự_đoán_VND": rec.get('Giá dự đoán'),
                            "Residual": rec.get('Residual'),
                            "Residual_z": rec.get('Residual_z'),
                            "Trạng_thái": "deleted",
                            "Thời_gian": datetime.now().isoformat()
                        }
                        aq = pd.concat([aq, pd.DataFrame([newrow])], ignore_index=True, sort=False)
                    cnt += 1
                try:
                    if not aq.empty:
                        aq.to_csv(ADMIN_PATH, index=False, encoding='utf-8')
                except Exception as e:
                    st.error(f"Lỗi khi lưu admin queue: {e}")

                # remove deleted from df_abnormal in session_state so UI cập nhật
                cur_df = st.session_state.get('df_abnormal', pd.DataFrame()).copy()
                st.session_state['df_abnormal'] = cur_df.loc[~cur_df['Href'].astype(str).isin(selected)].reset_index(drop=True)

                st.success(f"Đã đánh dấu XÓA cho {cnt} tin (cập nhật {ADMIN_PATH}).")
                for href in selected:
                    st.session_state['admin_selected'][href] = False
                st.rerun()

    # Approve ALL
    with colC:
        if st.button("✅ Duyệt TẤT CẢ"):
            hrefs = df_abnormal['Href'].astype(str).tolist()
            if not hrefs:
                st.info("Không có tin để duyệt.")
            else:
                if os.path.exists(UPDATED_PATH):
                    try:
                        df_upd = pd.read_csv(UPDATED_PATH, dtype=str)
                    except Exception:
                        df_upd = pd.DataFrame()
                else:
                    df_upd = pd.DataFrame()
                for href in hrefs:
                    rows = df_abnormal[df_abnormal['Href'].astype(str) == href]
                    if rows.empty:
                        continue
                    rec = rows.iloc[0].copy()
                    rec['Trạng_thái'] = 'approved'
                    rec['Thời_gian'] = datetime.now().isoformat()
                    rec_df = pd.DataFrame([rec])
                    if ('Href' in df_upd.columns) and ((df_upd['Href'].astype(str) == href).any()):
                        df_upd.loc[df_upd['Href'].astype(str) == href, rec_df.columns] = rec_df.iloc[0].values
                    else:
                        df_upd = pd.concat([df_upd, rec_df], ignore_index=True, sort=False)
                try:
                    df_upd.to_csv(UPDATED_PATH, index=False, encoding='utf-8')
                except Exception as e:
                    st.error(f"Lỗi khi lưu file cập nhật: {e}")
                if os.path.exists(ADMIN_PATH):
                    try:
                        aq = pd.read_csv(ADMIN_PATH, dtype=str)
                        if 'Href' in aq.columns:
                            aq.loc[aq['Href'].astype(str).isin(hrefs), 'Trạng_thái'] = 'approved'
                            aq.to_csv(ADMIN_PATH, index=False, encoding='utf-8')
                    except Exception:
                        pass

                # clear df_abnormal (approve all -> không còn tin bất thường)
                st.session_state['df_abnormal'] = pd.DataFrame()
                st.success(f"Đã duyệt toàn bộ {len(hrefs)} tin.")
                st.rerun()

    # Delete ALL
    with colD:
        if st.button("🗑️ Xóa TẤT CẢ"):
            hrefs = df_abnormal['Href'].astype(str).tolist()
            if not hrefs:
                st.info("Không có tin để xóa.")
            else:
                if os.path.exists(ADMIN_PATH):
                    try:
                        aq = pd.read_csv(ADMIN_PATH, dtype=str)
                    except Exception:
                        aq = pd.DataFrame()
                else:
                    aq = pd.DataFrame()
                for href in hrefs:
                    rows = df_abnormal[df_abnormal['Href'].astype(str) == href]
                    if rows.empty:
                        continue
                    rec = rows.iloc[0]
                    if not aq.empty and 'Href' in aq.columns and (aq['Href'].astype(str) == href).any():
                        aq.loc[aq['Href'].astype(str) == href, 'Trạng_thái'] = 'deleted'
                    else:
                        newrow = {
                            "Href": href,
                            "Thương hiệu": rec.get('Thương hiệu'),
                            "Dòng xe": rec.get('Dòng xe'),
                            "Giá_thực_VND": rec.get('Giá'),
                            "Giá_dự_đoán_VND": rec.get('Giá dự đoán'),
                            "Residual": rec.get('Residual'),
                            "Residual_z": rec.get('Residual_z'),
                            "Trạng_thái": "deleted",
                            "Thời_gian": datetime.now().isoformat()
                        }
                        aq = pd.concat([aq, pd.DataFrame([newrow])], ignore_index=True, sort=False)
                try:
                    if not aq.empty:
                        aq.to_csv(ADMIN_PATH, index=False, encoding='utf-8')
                except Exception as e:
                    st.error(f"Lỗi khi lưu admin queue: {e}")

                # clear df_abnormal
                st.session_state['df_abnormal'] = pd.DataFrame()
                st.success(f"Đã đánh dấu XÓA cho toàn bộ {len(hrefs)} tin.")
                st.rerun()

    # ---------- Pagination controls ----------
    st.markdown("---")
    pcol1, pcol2, pcol3 = st.columns([1,1,1])
    with pcol1:
        if st.button("← Trang trước") and st.session_state['abn_page'] > 0:
            st.session_state['abn_page'] -= 1
            st.rerun()
    with pcol2:
        st.markdown(f"Trang **{page+1}** / **{total_pages}**  —  Tổng: {total} tin")
    with pcol3:
        if st.button("Trang sau →") and st.session_state['abn_page'] < total_pages - 1:
            st.session_state['abn_page'] += 1
            st.rerun()

    # ---------- CSV EXPORT ----------
    csv = df_abnormal.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇ Tải CSV danh sách bất thường (toàn bộ)", data=csv, file_name="xe_bat_thuong.csv", mime="text/csv")