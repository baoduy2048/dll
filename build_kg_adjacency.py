import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "./data"
OUTPUT_DIR = "./data"
PPI_FILE = os.path.join(OUTPUT_DIR, "ppi_network.txt")
THRESHOLDS = {'gene': 0.7, 'methy': 0.75, 'mirna': 0.6}

def run():
    print("--- BƯỚC 3: XÂY DỰNG ĐỒ THỊ LAI GHÉP ---")
    
    # 1. Load Data
    try:
        with open(os.path.join(DATA_DIR, "feature_counts.txt")) as f:
            meta = {l.split('=')[0]: int(l.split('=')[1]) for l in f if '=' in l and 'classes' not in l}
        
        train_X = pd.read_csv(os.path.join(DATA_DIR, "train_X.csv"))
        test_X = pd.read_csv(os.path.join(DATA_DIR, "test_X.csv"))
        full_X = pd.concat([train_X, test_X], axis=0).reset_index(drop=True)
    except Exception as e:
        print(f"LỖI load dữ liệu: {e}")
        return

    # Tách views
    i1 = meta['num_gene']
    i2 = i1 + meta['num_cpg']
    X_gene = full_X.iloc[:, :i1]
    X_methy = full_X.iloc[:, i1:i2]
    X_mirna = full_X.iloc[:, i2:]

    # 2. Load PPI (Knowledge)
    ppi_mat = np.eye(X_gene.shape[1], dtype=np.float32)
    if os.path.exists(PPI_FILE):
        print("-> Đang tích hợp tri thức từ PPI Network...")
        gene_cols = list(X_gene.columns)
        gene_map = {name: i for i, name in enumerate(gene_cols)}
        
        with open(PPI_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    g1, g2 = parts[0], parts[1]
                    if g1 in gene_map and g2 in gene_map:
                        idx1, idx2 = gene_map[g1], gene_map[g2]
                        ppi_mat[idx1, idx2] = 1.0
                        ppi_mat[idx2, idx1] = 1.0
    else:
        print("-> Cảnh báo: Không thấy file PPI. Chạy ở chế độ không có tri thức.")

    # 3. Tính tương đồng & Lưu cạnh
    def save_edge(features, kg_mat, name, thresh):
        print(f"   Xử lý {name} view...")
        feat = features.values
        if kg_mat is not None:
            # Lan truyền tín hiệu tri thức
            feat = np.dot(feat, kg_mat)
        
        sim = cosine_similarity(feat)
        np.fill_diagonal(sim, 0)
        adj = np.where(sim > thresh, 1, 0)
        edges = sp.coo_matrix(adj)
        
        with open(os.path.join(OUTPUT_DIR, f"edges_{name}_gbm.csv"), 'w') as f:
            for s, d in zip(edges.row, edges.col):
                f.write(f"{s},{d}\n")
        print(f"   -> Đã lưu {edges.nnz} cạnh.")

    save_edge(X_gene, ppi_mat, "gene", THRESHOLDS['gene'])
    save_edge(X_methy, None, "methy", THRESHOLDS['methy'])
    save_edge(X_mirna, None, "mirna", THRESHOLDS['mirna'])
    print("-> Hoàn tất Bước 3.")

if __name__ == "__main__":
    run()