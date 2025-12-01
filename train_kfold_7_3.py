import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit 
from torch_geometric.data import Data
from model.heco import HeCo
from model.contrast import Contrast

# --- CẤU HÌNH ---
DATA_DIR = "./data"
EDGE_DIR = "./data"
RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CẤU HÌNH HUẤN LUYỆN ---
N_RUNS = 5        
TEST_SIZE = 0.3   
EPOCHS = 200      
HIDDEN_DIM = 256  

def load_full_data():
    """Load và gộp toàn bộ dữ liệu"""
    print("-> Đang tải toàn bộ dữ liệu...")
    
    meta = {}
    with open(os.path.join(DATA_DIR, "feature_counts.txt"), "r") as f:
        for l in f:
            if '=' in l:
                k, v = l.strip().split('=')
                if k!='classes': meta[k]=int(v)

    try:
        train_X = pd.read_csv(os.path.join(DATA_DIR, "train_X.csv"))
        test_X = pd.read_csv(os.path.join(DATA_DIR, "test_X.csv"))
        train_Y = pd.read_csv(os.path.join(DATA_DIR, "train_Y.csv"))
        test_Y = pd.read_csv(os.path.join(DATA_DIR, "test_Y.csv"))
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu.")
        return None

    full_X = pd.concat([train_X, test_X], axis=0).reset_index(drop=True)
    full_Y = pd.concat([train_Y, test_Y], axis=0).reset_index(drop=True)
    
    labels = torch.LongTensor(np.argmax(full_Y.values, axis=1)).to(DEVICE)
    labels_np = np.argmax(full_Y.values, axis=1) 

    i1 = meta['num_gene']
    i2 = i1 + meta['num_cpg']
    
    # Biến f1, f2, f3 chứa Tensor dữ liệu
    f1 = torch.FloatTensor(full_X.iloc[:, :i1].values).to(DEVICE)
    f2 = torch.FloatTensor(full_X.iloc[:, i1:i2].values).to(DEVICE)
    f3 = torch.FloatTensor(full_X.iloc[:, i2:].values).to(DEVICE)

    return f1, f2, f3, labels, labels_np

def load_edges(name):
    path = os.path.join(EDGE_DIR, name)
    edges = []
    with open(path, 'r') as f:
        for l in f:
            s, d = l.strip().split(',')
            edges.append([int(float(s)), int(float(d))])
    return torch.LongTensor(edges).T.to(DEVICE)

def main():
    print(f"--- BẮT ĐẦU HUẤN LUYỆN: {N_RUNS} LẦN (TỶ LỆ 7:3) ---")
    
    data_pack = load_full_data()
    if data_pack is None: return
    f1, f2, f3, labels, labels_np_for_split = data_pack # f1 ở đây là Tensor

    try:
        e1 = load_edges("edges_gene_gbm.csv")
        e2 = load_edges("edges_methy_gbm.csv")
        e3 = load_edges("edges_mirna_gbm.csv")
    except FileNotFoundError:
        print("LỖI: Thiếu file cạnh. Hãy chạy Bước 3.")
        return

    d1 = Data(x=f1, edge_index=e1, y=labels)
    d2 = Data(x=f2, edge_index=e2, y=labels)
    d3 = Data(x=f3, edge_index=e3, y=labels)

    sss = StratifiedShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=42)
    
    run_metrics = {'acc': [], 'f1': [], 'mcc': []}

    for run_idx, (train_idx_np, test_idx_np) in enumerate(sss.split(np.zeros(len(labels)), labels_np_for_split)):
        print(f"\n>>> Running Split {run_idx + 1}/{N_RUNS} (Train: {len(train_idx_np)} | Test: {len(test_idx_np)})")
        
        train_mask = torch.tensor(train_idx_np, dtype=torch.long).to(DEVICE)
        test_mask = torch.tensor(test_idx_np, dtype=torch.long).to(DEVICE)

        # Tại đây f1 vẫn phải là Tensor để lấy .shape
        model = HeCo(f1.shape[1], f2.shape[1], f3.shape[1], hidden_dim=HIDDEN_DIM).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        crit = Contrast(hidden_dim=HIDDEN_DIM, tau=0.5, lam=0.5).to(DEVICE)

        y_train = labels[train_mask]
        pos_mask = (y_train.unsqueeze(0) == y_train.unsqueeze(1)).float().to(DEVICE)

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            z1, z2, z3 = model(d1, d2, d3)
            loss = crit(z1[train_mask], z2[train_mask], z3[train_mask], pos_mask)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: Loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            embeds = model.get_embeds(d1, d2, d3)
            X_tr = embeds[train_mask].cpu().numpy()
            X_te = embeds[test_mask].cpu().numpy()
            y_tr = labels[train_mask].cpu().numpy()
            y_te = labels[test_mask].cpu().numpy()

            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            
            acc = accuracy_score(y_te, pred)
            
            # --- SỬA LỖI Ở ĐÂY: Đổi tên biến f1 thành f1_val ---
            f1_val = f1_score(y_te, pred, average='weighted') 
            mcc = matthews_corrcoef(y_te, pred)
            
            print(f"    -> Run {run_idx+1} Result: Accuracy={acc:.4f} | F1={f1_val:.4f}")
            
            run_metrics['acc'].append(acc)
            run_metrics['f1'].append(f1_val) # Append biến mới
            run_metrics['mcc'].append(mcc)

    print("\n" + "="*40)
    print(f"KẾT QUẢ TRUNG BÌNH ({N_RUNS} lần chạy - Tỷ lệ 7:3):")
    acc_mean, acc_std = np.mean(run_metrics['acc']), np.std(run_metrics['acc'])
    f1_mean = np.mean(run_metrics['f1'])
    mcc_mean = np.mean(run_metrics['mcc'])
    
    print(f"Accuracy : {acc_mean:.4f} (+/- {acc_std:.4f})")
    print(f"F1-Score : {f1_mean:.4f}")
    print(f"MCC      : {mcc_mean:.4f}")
    print("="*40)

    with open(os.path.join(RESULT_DIR, "split_7_3_metrics.txt"), "w") as f:
        f.write(f"Repeated Split Results (Runs={N_RUNS}, Train=0.7, Test=0.3)\n")
        f.write(f"Hidden Dim: {HIDDEN_DIM}\n")
        f.write(f"Accuracy: {acc_mean:.4f} +/- {acc_std:.4f}\n")
        f.write(f"F1: {f1_mean:.4f}\n")
        f.write(f"MCC: {mcc_mean:.4f}\n")

if __name__ == "__main__":
    main()