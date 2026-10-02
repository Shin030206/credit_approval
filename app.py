import urllib.parse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine
import streamlit as st

st.set_page_config(
    page_title="Credit Approval Predictor", page_icon="💳", layout="centered"
)
st.title("💳 Hệ Thống Dự Đoán Xét Duyệt Thẻ Tín Dụng")


# Huấn luyện mô hình trực tiếp từ MySQL (chỉ chạy 1 lần duy nhất khi mở app nhờ cache)
@st.cache_resource
def get_model():
  DB_USER = "avnadmin"
  DB_PASS = urllib.parse.quote_plus("MẬT_KHẨU_AIVEN_CỦA_BẠN")
  DB_HOST = "mysql-xxxxx.a.aivencloud.com"
  DB_PORT = "12345"  # Xem đúng cổng trên Aiven
  DB_NAME = "defaultdb"

  uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
  engine = create_engine(uri)

  data = pd.read_sql("SELECT * FROM credit_approval;", con=engine)
  data = data.drop(columns=["id"], errors="ignore")

  X = data.drop(columns=["approval_status"])
  y = data["approval_status"].map({"+": 1, "-": 0})

  num_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
  cat_cols = [c for c in X.columns if c not in num_cols]

  num_pipe = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
  ])
  cat_pipe = Pipeline([
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("ohe", OneHotEncoder(handle_unknown="ignore")),
  ])

  prep = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
  clf = Pipeline([
      ("prep", prep),
      ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
  ])

  clf.fit(X, y)
  return clf


with st.spinner("Đang kết nối database và khởi tạo mô hình..."):
  model = get_model()

# Giao diện Form nhập thông tin
with st.form("credit_form"):
  st.subheader("Thông tin hồ sơ khách hàng")
  col1, col2 = st.columns(2)

  with col1:
    a1 = st.selectbox("A1 (Giới tính / Nhóm)", ["a", "b"])
    a2 = st.number_input("A2 (Tuổi)", min_value=15.0, max_value=90.0, value=30.0)
    a3 = st.number_input("A3 (Nợ hiện tại)", min_value=0.0, value=2.5)
    a4 = st.selectbox("A4 (Tình trạng hôn nhân)", ["u", "y", "l", "t"])
    a5 = st.selectbox("A5 (Loại khách hàng)", ["g", "p", "gg"])
    a6 = st.selectbox(
        "A6 (Ngành nghề)",
        [
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
        "A7 (Dân tộc / Nhóm)",
        ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"],
    )

  with col2:
    a8 = st.number_input("A8 (Số năm kinh nghiệm)", min_value=0.0, value=1.5)
    a9 = st.selectbox(
        "A9 (Lịch sử nợ xấu)", ["t", "f"], help="t: Có nợ xấu, f: Không"
    )
    a10 = st.selectbox("A10 (Tình trạng việc làm)", ["t", "f"])
    a11 = st.number_input(
        "A11 (Điểm tín dụng)", min_value=0, max_value=70, value=2
    )
    a12 = st.selectbox("A12 (Giấy phép lái xe)", ["t", "f"])
    a13 = st.selectbox("A13 (Tình trạng cư trú)", ["g", "p", "s"])
    a14 = st.number_input("A14 (Mã bưu chính)", min_value=0.0, value=100.0)
    a15 = st.number_input("A15 (Thu nhập)", min_value=0, value=500)

  submit = st.form_submit_button("🔍 Tiến hành Xét Duyệt")

if submit:
  input_df = pd.DataFrame(
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
  pred = model.predict(input_df)[0]
  prob = model.predict_proba(input_df)[0][1]

  st.divider()
  if pred == 1:
    st.success(f"✅ **KẾT QUẢ: ĐƯỢC DUYỆT CẤP THẺ** (Xác suất: {prob*100:.1f}%)")
  else:
    st.error(
        f"❌ **KẾT QUẢ: TỪ CHỐI DUYỆT** (Xác suất duyệt: {prob*100:.1f}%)"
    )
