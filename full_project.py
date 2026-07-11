# =====================================================================================
# ĐỀ TÀI: DỰ ĐOÁN SỐ ĐƠN HÀNG SHOPEEFOOD TẠI TP.HCM
# FILE: full_project.py — CODE ĐÃ SỬA LẠI KHỚP ĐÚNG FILE shopeefood_daily.csv THẬT
# =====================================================================================
# Cấu trúc dữ liệu thật (500 dòng, 11 cột):
#   ngay                  : ngày (dạng yyyy-mm-dd)
#   ten_quan              : tên quán/cửa hàng
#   danh_muc              : danh mục món ăn (Cơm, Bún/Phở, Trà sữa,...)
#   khu_vuc               : quận/huyện (Quận 1, Quận 7, Thủ Đức,...)
#   so_don_hang           : số đơn hàng  -> BIẾN MỤC TIÊU (target)
#   gia_tri_don_tb_vnd    : giá trị đơn trung bình (VNĐ)
#   doanh_thu_vnd         : doanh thu (VNĐ)
#   danh_gia_sao          : đánh giá sao (3.5 - 5.0)
#   thoi_gian_giao_phut   : thời gian giao hàng (phút)
#   khuyen_mai            : loại khuyến mãi (Freeship, Giảm 50%,...)
#   ty_le_huy_don_%       : tỷ lệ hủy đơn (%)
#
# Quy trình 4 bước:
#   BƯỚC 1: Đọc dữ liệu, làm sạch, xử lý outlier, vẽ biểu đồ khám phá dữ liệu
#   BƯỚC 2: Tạo biến mới (feature engineering) từ cột ngày
#   BƯỚC 3: Chọn thuật toán, huấn luyện (train), tinh chỉnh tham số   <-- PHẦN THUẬT TOÁN
#   BƯỚC 4: Đánh giá sai số (MAE, RMSE)
# =====================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# XGBoost là thư viện ngoài, cần cài: pip install xgboost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Chưa cài xgboost -> bỏ qua mô hình XGBoost. Cài bằng: pip install xgboost")

plt.rcParams["font.family"] = "sans-serif"  # tránh lỗi hiển thị tiếng Việt trên biểu đồ


# =====================================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU + KHÁM PHÁ BAN ĐẦU
# =====================================================================================
DATA_PATH = "shopeefood_daily.csv"   # đổi lại đường dẫn file của bạn nếu cần

df = pd.read_csv(DATA_PATH, parse_dates=["ngay"])

print("===== 5 dòng đầu =====")
print(df.head())

print("\n===== Thông tin dữ liệu =====")
print(df.info())

print("\n===== Thống kê mô tả =====")
print(df.describe())

# -------------------------------------------------------------------------------------
# 1.1. Kiểm tra & xử lý giá trị thiếu, trùng lặp
# -------------------------------------------------------------------------------------
print("\n===== Số giá trị thiếu theo cột =====")
print(df.isna().sum())

print("\nSố dòng trùng lặp:", df.duplicated().sum())

for col in df.columns:
    if df[col].isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

# -------------------------------------------------------------------------------------
# 1.2. Xử lý giá trị không hợp lệ
# -------------------------------------------------------------------------------------
# so_don_hang không được âm
df = df[df["so_don_hang"] >= 0]

# ty_le_huy_don_% phải nằm trong [0, 100]
df = df[(df["ty_le_huy_don_%"] >= 0) & (df["ty_le_huy_don_%"] <= 100)]

# danh_gia_sao phải nằm trong [0, 5]
df = df[(df["danh_gia_sao"] >= 0) & (df["danh_gia_sao"] <= 5)]

# -------------------------------------------------------------------------------------
# 1.3. Xử lý outlier bằng phương pháp IQR (giữ nguyên, chỉ đánh dấu để tham khảo)
# -------------------------------------------------------------------------------------
Q1 = df["so_don_hang"].quantile(0.25)
Q3 = df["so_don_hang"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["so_don_hang"] < lower_bound) | (df["so_don_hang"] > upper_bound)]
print(f"\nSố dòng outlier theo so_don_hang: {len(outliers)}")
if len(outliers) > 0:
    print("Các quán có outlier:", outliers["ten_quan"].unique())
# -> Nhóm quyết định GIỮ NGUYÊN các outlier này vì phản ánh đúng thực tế kinh doanh

# -------------------------------------------------------------------------------------
# 1.4. Trực quan hóa - vẽ biểu đồ phân tích xu hướng đơn hàng
# -------------------------------------------------------------------------------------
# Hình 1: Top 10 khu vực có số đơn hàng trung bình cao nhất
plt.figure(figsize=(10, 6))
top10 = df.groupby("khu_vuc")["so_don_hang"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top10.values, y=top10.index, palette="viridis")
plt.title("Top 10 khu vực có số đơn hàng trung bình cao nhất")
plt.xlabel("Số đơn hàng trung bình")
plt.ylabel("Khu vực")
plt.tight_layout()
plt.savefig("hinh1_top10_khu_vuc.png")
plt.close()

# Hình 2: Phân phối số đơn hàng theo danh mục món ăn
plt.figure(figsize=(10, 6))
sns.boxplot(x="so_don_hang", y="danh_muc", data=df)
plt.title("Phân phối số đơn hàng theo danh mục món ăn")
plt.tight_layout()
plt.savefig("hinh2_danh_muc.png")
plt.close()

# Hình 3: Xu hướng số đơn hàng trung bình theo tháng
df["thang"] = df["ngay"].dt.month
plt.figure(figsize=(8, 5))
don_theo_thang = df.groupby("thang")["so_don_hang"].mean()
sns.lineplot(x=don_theo_thang.index, y=don_theo_thang.values, marker="o")
plt.title("Số đơn hàng trung bình theo tháng")
plt.xlabel("Tháng")
plt.ylabel("Số đơn hàng trung bình")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("hinh3_xu_huong_theo_thang.png")
plt.close()

# Hình 4: Ma trận tương quan giữa các biến số
plt.figure(figsize=(9, 7))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Ma trận tương quan giữa các biến số")
plt.tight_layout()
plt.savefig("hinh4_ma_tran_tuong_quan.png")
plt.close()

print("\nĐã lưu 4 biểu đồ: hinh1_..., hinh2_..., hinh3_..., hinh4_...png")


# =====================================================================================
# BƯỚC 2: TẠO CÁC BIẾN MỚI (FEATURE ENGINEERING)
# =====================================================================================
# Thu: thứ trong tuần (0 = Thứ Hai ... 6 = Chủ Nhật)
df["thu"] = df["ngay"].dt.dayofweek

# Cuoi_tuan: 1 nếu Thứ Bảy/Chủ Nhật, 0 nếu ngày thường
df["cuoi_tuan"] = df["thu"].apply(lambda x: 1 if x >= 5 else 0)

# ngay_trong_thang: ngày trong tháng (1 - 31)
df["ngay_trong_thang"] = df["ngay"].dt.day

# thang: tháng trong năm (đã tạo ở bước vẽ biểu đồ, giữ lại làm đặc trưng)
# (cột "thang" đã có sẵn từ bước 1.4)

print("\n===== Sau khi tạo biến mới =====")
print(df[["ngay", "thu", "cuoi_tuan", "ngay_trong_thang", "thang"]].head())

df.to_csv("Cleaned_Data.csv", index=False)
print("\nĐã lưu dữ liệu sạch vào Cleaned_Data.csv")


# #####################################################################################
# #####################################################################################
# BƯỚC 3: CHỌN THUẬT TOÁN, HUẤN LUYỆN (TRAIN) VÀ TINH CHỈNH THAM SỐ MÔ HÌNH
# (>>> ĐÂY LÀ PHẦN THUẬT TOÁN RIÊNG CỦA BẠN <<<)
# #####################################################################################
# #####################################################################################

# -------------------------------------------------------------------------------------
# 3.1. Chọn biến đầu vào (features) và biến mục tiêu (target)
# -------------------------------------------------------------------------------------
target_col = "so_don_hang"

feature_cols_numeric = [
    "gia_tri_don_tb_vnd", "doanh_thu_vnd", "danh_gia_sao",
    "thoi_gian_giao_phut", "ty_le_huy_don_%",
    "thu", "cuoi_tuan", "ngay_trong_thang", "thang"
]
feature_cols_categorical = ["khu_vuc", "danh_muc", "khuyen_mai"]

df_model = df.copy()
label_encoders = {}
for col in feature_cols_categorical:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

feature_cols = feature_cols_numeric + feature_cols_categorical

X = df_model[feature_cols]
y = df_model[target_col]

# -------------------------------------------------------------------------------------
# 3.2. Chia tập train / test
# -------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------------------------------------------------------------
# 3.3. Hàm đánh giá sai số (MAE, RMSE) — dùng chung cho BƯỚC 4
# -------------------------------------------------------------------------------------
def danh_gia(y_true, y_pred, ten_mo_hinh):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{ten_mo_hinh:25s} | MAE: {mae:10.2f} | RMSE: {rmse:10.2f}")
    return mae, rmse


ket_qua = []

# -------------------------------------------------------------------------------------
# 3.4. Thuật toán 1: Linear Regression
# -------------------------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mae, rmse = danh_gia(y_test, y_pred_lr, "Linear Regression")
ket_qua.append(("Linear Regression", mae, rmse))

# -------------------------------------------------------------------------------------
# 3.5. Thuật toán 2: Random Forest Regressor (tinh chỉnh tham số bằng GridSearchCV)
# -------------------------------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)
rf_grid.fit(X_train, y_train)  # Random Forest không cần chuẩn hóa dữ liệu

best_rf = rf_grid.best_estimator_
print("\nTham số tốt nhất cho Random Forest:", rf_grid.best_params_)

y_pred_rf = best_rf.predict(X_test)
mae, rmse = danh_gia(y_test, y_pred_rf, "Random Forest (tuned)")
ket_qua.append(("Random Forest (tuned)", mae, rmse))

# -------------------------------------------------------------------------------------
# 3.6. Thuật toán 3: XGBoost Regressor (tinh chỉnh tham số bằng GridSearchCV)
# -------------------------------------------------------------------------------------
if HAS_XGB:
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }

    xgb_grid = GridSearchCV(
        estimator=XGBRegressor(random_state=42, objective="reg:squarederror"),
        param_grid=xgb_param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)

    best_xgb = xgb_grid.best_estimator_
    print("\nTham số tốt nhất cho XGBoost:", xgb_grid.best_params_)

    y_pred_xgb = best_xgb.predict(X_test)
    mae, rmse = danh_gia(y_test, y_pred_xgb, "XGBoost (tuned)")
    ket_qua.append(("XGBoost (tuned)", mae, rmse))

# #####################################################################################
# KẾT THÚC PHẦN THUẬT TOÁN
# #####################################################################################


# =====================================================================================
# BƯỚC 4: ĐÁNH GIÁ SAI SỐ TỔNG HỢP (MAE, RMSE) GIỮA CÁC MÔ HÌNH
# =====================================================================================
print("\n===== BẢNG SO SÁNH CÁC MÔ HÌNH =====")
df_ket_qua = pd.DataFrame(ket_qua, columns=["Mo_Hinh", "MAE", "RMSE"])
df_ket_qua = df_ket_qua.sort_values("RMSE").reset_index(drop=True)
print(df_ket_qua)

mo_hinh_tot_nhat = df_ket_qua.iloc[0]["Mo_Hinh"]
print(f"\n=> Mô hình có sai số thấp nhất (RMSE nhỏ nhất): {mo_hinh_tot_nhat}")

df_ket_qua.to_csv("ket_qua_danh_gia_mo_hinh.csv", index=False)
print("Đã lưu bảng so sánh vào ket_qua_danh_gia_mo_hinh.csv")