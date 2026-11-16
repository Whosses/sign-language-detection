# BÀN GIAO DỮ LIỆU CHO THÀNH VIÊN 2 (Data & Preprocessing -> Model)

## 1. Tổng quan dataset
- Bài toán: nhận dạng ký hiệu tay tĩnh (ASL), 10 lớp: A, B, C, D, E, F, I, L, V, W
- Nguồn: tự thu thập bằng webcam, đợt s1 (thông tin điều kiện chụp: [điền: số người, tay trái/phải, nền, ánh sáng])
- Số ảnh gốc: 1000 (100 ảnh/lớp). Ảnh MediaPipe không nhận ra tay: 16 (1,6%). Còn lại: 984 mẫu
- Sau làm sạch: không thiếu giá trị, không trùng, không hỏng. Số mẫu mỗi lớp: A 100, B 98, C 93, D 100, E 100, F 100, I 100, L 100, V 93, W 100

## 2. Định dạng feature
- Mỗi mẫu = 21 landmark bàn tay (MediaPipe Hands) x (x, y, z) = 63 số
- Tên cột: x1, y1, z1, ..., x21, y21, z21, label
- Chuẩn hóa: lấy cổ tay (landmark 1) làm gốc tọa độ, rồi chia cho khoảng cách 2D xa nhất tới cổ tay. Nhờ đó không phụ thuộc vị trí và kích thước tay trong ảnh
- Lưu ý: 3 cột x1, y1, z1 (cổ tay) luôn bằng 0 sau chuẩn hóa (có thể bỏ khi train)

## 3. File bàn giao (thư mục `splits/`)
- `X_train.npy`, `y_train.npy`: 688 mẫu
- `X_val.npy`, `y_val.npy`: 148 mẫu
- `X_test.npy`, `y_test.npy`: 148 mẫu
- `train.csv`, `val.csv`, `test.csv`: cùng dữ liệu dạng CSV (có cột label chữ)
- `label_map.json`: {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "I": 6, "L": 7, "V": 8, "W": 9}
- `dataset.csv`: toàn bộ 984 mẫu đã chuẩn hóa
- Hình EDA: thư mục `eda_output/` (đếm mẫu, heatmap, PCA, t-SNE, skeleton)

## 4. Cách tách tập
- Với mỗi lớp, chia theo thứ tự chụp: 70% đầu = train, 15% kế = val, 15% cuối = test
- Cùng một đợt chụp (cùng nền, ánh sáng), nên độ chính xác test có thể cao hơn thực tế. Khi có đợt s2 sẽ dùng làm test độc lập

## 5. Nhận xét từ EDA (để chọn model và đánh giá)
- t-SNE tách 10 lớp thành các cụm riêng rõ
- Các lớp nắm tay (A, E, I) và cặp V/W, D/F là những cặp cần theo dõi trong confusion matrix
- Lớp C và D có nhiều cụm con (tư thế/hướng tay chưa hoàn toàn nhất quán trong lúc chụp)
- Gợi ý model thử: kNN, SVM, Random Forest, MLP nhỏ (không cần GPU)

## 6. Cách nạp dữ liệu
```python
import numpy as np, json
X_train = np.load("splits/X_train.npy"); y_train = np.load("splits/y_train.npy")
X_val = np.load("splits/X_val.npy");     y_val = np.load("splits/y_val.npy")
X_test = np.load("splits/X_test.npy");   y_test = np.load("splits/y_test.npy")
label_map = json.load(open("splits/label_map.json"))
```

## 7. Tái hiện pipeline
python capture.py <lớp> <số ảnh> <đợt>  ->  python extract_landmarks.py  ->  python preprocess.py  ->  python eda.py  ->  python split.py
