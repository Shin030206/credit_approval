import urllib.parse
import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine
import streamlit as st

st.set_page_config(
    page_title="Credit Approval System", page_icon="💳", layout="wide"
)
st.title("💳 Hệ Thống Phân Tích & Dự Đoán Xét Duyệt Thẻ Tín Dụng")


# 1. Kết nối DB và Load dữ liệu (Được cache để chạy cực nhanh)
@st.cache_resource
def load_data_and_model():
  db_user = "avnadmin"
  db_pass = urllib.parse.quote_plus("AVNS_vXMqtHh9OnEtJYzTI_b")
  db_host = "mysql-1670de5d-khanhnguyenngoc82-a7fc.d.aivencloud.com"
  db_port = "11821"  # Xem đúng cổng trên Aiven
  db_name = "defaultdb"

  def connect_db():
    return pymysql.connect(
        host=db_host,
        port=int(db_port),
        user=db_user,
        password=db_pass.strip(),
        database=db_name,
        charset="utf8mb4",
    )

  engine = create_engine("mysql+pymysql://", creator=connect_db)
  df = pd.read_sql("SELECT * FROM credit_approval;", con=engine)
  df = df.drop(columns=["id"], errors="ignore")

  # Xây dựng Pipeline Model
  X = df.drop(columns=["approval_status"])
  y = df["approval_status"].map({"+": 1, "-": 0})

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

  return df, clf


with st.spinner("Đang tải dữ liệu từ MySQL Cloud..."):
  df, model = load_data_and_model()

# 2. Tạo 2 Tabs: Trực quan hóa & Dự đoán
tab1, tab2 = st.tabs(
    ["📊 Trực quan hóa dữ liệu (Visualization)", "🔮 Dự đoán xét duyệt hồ sơ"]
)

# ================= TAB 1: DATA VISUALIZATION =================
with tab1:
  st.subheader("Khám phá và Trực quan hóa Dữ liệu Tín dụng")

  col_metric1, col_metric2, col_metric3 = st.columns(3)
  col_metric1.metric("Tổng số hồ sơ", len(df))
  col_metric2.metric("Số hồ sơ được duyệt (+)", (df["approval_status"] == "+").sum())
  col_metric3.metric("Số hồ sơ bị từ chối (-)", (df["approval_status"] == "-").sum())

  st.divider()

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  sns.set_theme(style="whitegrid")

  # Biểu đồ 1: Tỷ lệ duyệt
  sns.countplot(
      data=df,
      x="approval_status",
      ax=axes[0, 0],
      palette=["#2ecc71", "#e74c3c"],
  )
  axes[0, 0].set_title(
      "1. Tỷ lệ duyệt hồ sơ tín dụng (+: Duyệt, -: Từ chối)", weight="bold"
  )
  axes[0, 0].set_xlabel("Trạng thái")
  axes[0, 0].set_ylabel("Số lượng")

  # Biểu đồ 2: Phân phối điểm tín dụng A11
  sns.boxplot(
      data=df,
      x="approval_status",
      y="A11",
      ax=axes[0, 1],
      palette=["#2ecc71", "#e74c3c"],
      showfliers=False,
  )
  axes[0, 1].set_title(
      "2. Điểm tín dụng (A11) theo Nhóm duyệt", weight="bold"
  )
  axes[0, 1].set_xlabel("Trạng thái duyệt")
  axes[0, 1].set_ylabel("Điểm tín dụng (A11)")

  # Biểu đồ 3: Lịch sử nợ xấu A9
  cross_tab = (
      pd.crosstab(df["A9"], df["approval_status"], normalize="index") * 100
  )
  cross_tab.plot(
      kind="bar",
      stacked=True,
      ax=axes[1, 0],
      color=["#2ecc71", "#e74c3c"],
      rot=0,
  )
  axes[1, 0].set_title(
      "3. Tỷ lệ duyệt theo Lịch sử nợ xấu (A9: t=có, f=không)", weight="bold"
  )
  axes[1, 0].set_xlabel("Lịch sử nợ xấu (A9)")
  axes[1, 0].set_ylabel("Tỷ lệ (%)")

  # Biểu đồ 4: Heatmap tương quan
  numeric_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
  corr = df[numeric_cols].corr()
  sns.heatmap(
      corr,
      annot=True,
      cmap="coolwarm",
      fmt=".2f",
      ax=axes[1, 1],
      vmin=-1,
      vmax=1,
  )
  axes[1, 1].set_title(
      "4. Ma trận tương quan giữa các biến định lượng", weight="bold"
  )

  plt.tight_layout()
  st.pyplot(fig)

# ================= TAB 2: DỰ BÁO =================
with tab2:
  st.subheader("Nhập thông tin hồ sơ để dự báo")

  with st.form("credit_form"):
    c1, c2 = st.columns(2)
    with c1:
      a1 = st.selectbox("A1 (Giới tính / Nhóm)", ["a", "b"])
      a2 = st.number_input(
          "A2 (Tuổi)", min_value=15.0, max_value=90.0, value=30.0
      )
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

    with c2:
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
      st.success(
          f"✅ **KẾT QUẢ: ĐƯỢC DUYỆT CẤP THẺ** (Xác suất: {prob*100:.1f}%)"
      )
    else:
      st.error(
          f"❌ **KẾT QUẢ: TỪ CHỐI DUYỆT** (Xác suất duyệt: {prob*100:.1f}%)"
      )
