import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Thiết lập giao diện trang web
st.set_page_config(
    page_title="Credit Approval System", page_icon="💳", layout="centered"
)

st.title("💳 Hệ Thống Dự Đoán Xét Duyệt Thẻ Tín Dụng")
st.markdown(
    "Ứng dụng học máy dự đoán khả năng duyệt hồ sơ tín dụng dựa trên mô hình Random Forest."
)


# Load mô hình
@st.cache_resource
def load_model():
  return joblib.load("credit_model.pkl")


model = load_model()

# Form nhập liệu
with st.form("prediction_form"):
  st.subheader("Thông tin ứng viên")

  col1, col2 = st.columns(2)
  with col1:
    a1 = st.selectbox("Giới tính / Nhóm A1", options=["a", "b"])
    a2 = st.number_input(
        "Tuổi (A2)", min_value=18.0, max_value=85.0, value=30.0, step=1.0
    )
    a3 = st.number_input("Nợ hiện tại (A3)", min_value=0.0, value=2.5, step=0.1)
    a4 = st.selectbox("Tình trạng gia đình (A4)", options=["u", "y", "l", "t"])
    a5 = st.selectbox("Khách hàng (A5)", options=["g", "p", "gg"])
    a6 = st.selectbox(
        "Ngành nghề (A6)",
        options=[
            "c",
            "d",
            "cc",
            "i",
            "j",
            "k",
            "m",
            "r",
            "q",
            "w",
            "x",
            "e",
            "aa",
            "ff",
        ],
    )
    a7 = st.selectbox(
        "Dân tộc / Nhóm (A7)",
        options=["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"],
    )

  with col2:
    a8 = st.number_input(
        "Năm kinh nghiệm (A8)", min_value=0.0, value=1.5, step=0.1
    )
    a9 = st.selectbox(
        "Lịch sử nợ xấu (A9)",
        options=["t", "f"],
        help="t: Có vi phạm/nợ xấu, f: Không",
    )
    a10 = st.selectbox("Việc làm hiện tại (A10)", options=["t", "f"])
    a11 = st.number_input(
        "Điểm tín dụng (A11)", min_value=0, max_value=70, value=2, step=1
    )
    a12 = st.selectbox("Giấy phép lái xe (A12)", options=["t", "f"])
    a13 = st.selectbox("Tình trạng cư trú (A13)", options=["g", "p", "s"])
    a14 = st.number_input(
        "Mã bưu chính / Tài khoản (A14)", min_value=0.0, value=100.0, step=10.0
    )
    a15 = st.number_input(
        "Thu nhập (A15)", min_value=0, value=500, step=100
    )  # Thu nhập

  submit_btn = st.form_submit_button("🔍 Tiến hành Xét Duyệt")

if submit_btn:
  # Gom dữ liệu thành DataFrame
  input_data = pd.DataFrame(
      [[
          a1,
          a2,
          a3,
          a4,
          a5,
          a6,
          a7,
          a8,
          a9,
          a10,
          a11,
          a12,
          a13,
          a14,
          a15,
      ]],
      columns=[f"A{i}" for i in range(1, 16)],
  )

  # Dự đoán
  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1]

  st.divider()
  if prediction == 1:
    st.success(
        f"✅ **KẾT QUẢ: ĐƯỢC DUYỆT CẤP THẺ** (Độ tin cậy: {probability*100:.1f}%)"
    )
  else:
    st.error(
        f"❌ **KẾT QUẢ: TỪ CHỐI DUYỆT** (Độ tin cậy: {(1-probability)*100:.1f}%)"
    )
