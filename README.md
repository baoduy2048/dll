Bước 1: Tiền xử lý dữ liệu (Data Preprocessing)
Thực hiện lọc dữ liệu, tìm giao điểm các mẫu bệnh nhân chung và chuẩn hóa Z-score.

Bước 2: Tải Tri thức Sinh học (Fetch Knowledge)
Tự động gọi API của STRING DB để tải mạng lưới tương tác Protein-Protein (PPI) cho Top 2000 gen quan trọng nhất.
- Input: Danh sách gen từ train_X.csv.
- Output: data/ppi_network.txt.

Bước 3: Xây dựng Đồ thị Lai ghép (Hybrid Graph Construction)
Tạo ma trận kề (Adjacency Matrix) để nối các bệnh nhân lại với nhau.

- Input: Dữ liệu đã xử lý + ppi_network.txt.
- Output: data/edges_gene_gbm.csv, data/edges_methy_gbm.csv, edges_mirna_gbm.csv.

Bước 4: Huấn luyện & Đánh giá (Training & Evaluation)
Chạy mô hình GCN với cơ chế Supervised Contrastive Learning.
Cấu hình:
- Chạy lặp lại 5 lần.
- Tỷ lệ chia: Train 70% - Test 30% (Stratified Shuffle Split).
- Hidden Dimension: 256.
- Epochs: 150.
- Output: Độ chính xác (Accuracy), F1-Score
