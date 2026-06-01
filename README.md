# Dự báo Giá Cổ Phiếu VN30 – Machine Learning & Deep Learning

**Đề tài:** Dự báo giá cổ phiếu sử dụng công nghệ Machine Learning  
**Sinh viên:** Phạm Kim Vương (23AI057) & Đặng Khánh Dũng (23AI007)  
**GVHD:** TS. Đặng Đại Thọ

---

## Cấu trúc dự án

```
stock_prediction/
├── config.py            # Cấu hình (đường dẫn, tỷ lệ chia, hyperparameters)
├── data_processing.py   # Load, tiền xử lý, feature engineering, scale
├── models.py            # ARIMA, LinearRegression, RandomForest, LSTM, GRU
├── evaluation.py        # MAE/RMSE/MAPE, đồ thị, export CSV
├── main.py              # Pipeline chính
├── requirements.txt
│
├── VN30/
│   └── VN30_Data/
│       ├── ACB/
│       │   ├── ACB_2y.csv
│       │   ├── ACB_5y.csv
│       │   └── ACB_historical.csv   ← mặc định dùng file này
│       ├── VCB/  ...
│       └── (30 mã VN30)
│
├── models/              # Model LSTM/GRU lưu tự động (.keras)
└── results/             # Đồ thị (.png) và bảng kết quả (.csv)
```

---

## Cấu trúc file CSV (dữ liệu crawl)

| Cột    | Kiểu    | Mô tả                  |
|--------|---------|------------------------|
| Date   | string  | Ngày giao dịch         |
| Open   | float   | Giá mở cửa             |
| High   | float   | Giá cao nhất           |
| Low    | float   | Giá thấp nhất          |
| Close  | float   | Giá đóng cửa (mục tiêu)|
| Volume | int     | Khối lượng giao dịch   |

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Chạy chương trình

```bash
# Chạy toàn bộ 30 mã VN30 (file _historical.csv)
python main.py

# Chỉ một mã
python main.py --ticker VCB

# Dùng file 2 năm
python main.py --ticker VCB --mode 2y

# Bỏ qua LSTM/GRU (nhanh hơn, dùng khi test)
python main.py --ticker VCB --no-dl

# Chỉ vẽ EDA (không huấn luyện)
python main.py --eda-only
```

---

## Pipeline

```
Load  VN30/VN30_Data/<TICKER>/<TICKER>_historical.csv
        ↓
Tiền xử lý
  - Parse cột Date → datetime, sort tăng dần
  - Xử lý giá trị 0 / NaN (forward fill)
  - Loại duplicate
        ↓
Feature Engineering (30+ đặc trưng)
  ├── Lag features    : Close t-1, t-2, t-3, t-5, t-10
  ├── SMA / EMA       : 5, 10, 20, 50 phiên
  ├── RSI             : 14 phiên
  ├── Bollinger Bands : 20 phiên ±2σ
  ├── Daily Return, 5d Return, Volatility
  ├── High-Low ratio, Open-Close ratio
  └── Volume MA5, Volume ratio
        ↓
Chia dữ liệu (theo thời gian)
  Train 70%  │  Val 15%  │  Test 15%
        ↓
Chuẩn hóa MinMaxScaler (fit trên Train)
        ↓
Huấn luyện & Đánh giá
  ├── ARIMA(5,1,0)
  ├── Linear Regression
  ├── Random Forest (200 cây)
  ├── LSTM  [64→32] + Dropout(0.2)
  └── GRU   [64→32] + Dropout(0.2)
        ↓
So sánh MAE / RMSE / MAPE  →  lưu results/
```

---

## Kết quả đầu ra

Sau khi chạy, thư mục `results/` chứa:

| File | Nội dung |
|------|----------|
| `eda_<TICKER>.png` | Lịch sử giá + khối lượng |
| `predictions_<TICKER>.png` | Giá thực tế vs dự báo trên test |
| `metrics_<TICKER>.png` | Bar chart MAE/RMSE/MAPE |
| `metrics_<TICKER>.csv` | Bảng số liệu từng mô hình |
| `summary_all.csv` | Tổng hợp tất cả 30 mã |

---

## Điều chỉnh cấu hình (`config.py`)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `DATA_MODE` | `"historical"` | Loại file: `2y`, `5y`, `historical` |
| `ARIMA_ORDER` | `(5, 1, 0)` | Bậc (p, d, q) |
| `SEQUENCE_LEN` | `30` | Độ dài chuỗi LSTM/GRU |
| `LSTM_UNITS` | `[64, 32]` | Số unit mỗi lớp |
| `DL_EPOCHS` | `50` | Số epoch tối đa |
| `TRAIN_RATIO` | `0.70` | Tỷ lệ tập train |
