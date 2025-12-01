import pandas as pd
import requests
import os

# --- CẤU HÌNH ---
DATA_DIR = "./data"
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PPI_FILE = os.path.join(OUTPUT_DIR, "ppi_network.txt")

# URL API của STRING DB
STRING_API_URL = "https://string-db.org/api/tsv/network"

def get_gene_list():
    """Lấy danh sách gen từ file train_X.csv"""
    print("1. Đang đọc danh sách Gen từ dữ liệu huấn luyện...")
    
    # 1. Xác định số lượng Gen
    meta_path = os.path.join(DATA_DIR, "feature_counts.txt")
    num_gene = 0
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if line.startswith("num_gene="):
                    num_gene = int(line.strip().split('=')[1])
                    break
    
    # 2. Đọc tên Gen
    df = pd.read_csv(os.path.join(DATA_DIR, "train_X.csv"), nrows=1)
    
    if num_gene == 0:
        num_gene = 2000 # Fallback
        
    all_genes = df.columns[:num_gene].tolist()
    print(f"   -> Dữ liệu gốc có: {len(all_genes)} gen.")

    # --- SỬA LỖI: CẮT GIẢM XUỐNG 2000 ---
    if len(all_genes) > 2000:
        print("   [CẢNH BÁO] STRING DB giới hạn tối đa 2000 node.")
        print("   -> Đang chọn 2000 gen đầu tiên (Top features).")
        selected_genes = all_genes[:2000]
    else:
        selected_genes = all_genes

    return selected_genes

def fetch_from_string_db(genes):
    """Gửi yêu cầu lên STRING DB"""
    print(f"2. Đang gửi yêu cầu cho {len(genes)} gen đến STRING DB API...")
    
    params = {
        "identifiers": "%0d".join(genes),
        "species": 9606,  # Homo sapiens
        "caller_identity": "www.my_experiment.org"
    }

    try:
        response = requests.post(STRING_API_URL, data=params)
        
        if response.status_code == 200:
            print("   -> Kết nối thành công! Đang xử lý dữ liệu...")
            
            lines = response.text.splitlines()
            if len(lines) <= 1:
                print("   [Cảnh báo] STRING không tìm thấy tương tác nào cho danh sách này.")
                return

            # Lưu file sạch chỉ chứa 2 cột tên Gen để build_kg_adjacency dễ đọc
            with open(PPI_FILE, 'w', encoding='utf-8') as f:
                count = 0
                for line in lines:
                    # Bỏ qua header
                    if line.startswith("stringId") or line.startswith("#"):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        # Format trả về: stringId_A, stringId_B, preferredName_A, preferredName_B, ...
                        # Chúng ta lấy cột 2 và 3 (Tên Gen)
                        gene_a = parts[2]
                        gene_b = parts[3]
                        score = parts[5] if len(parts) > 5 else "0"
                        
                        # Chỉ lưu nếu score > 0.4 (độ tin cậy trung bình)
                        if float(score) > 0.4:
                            f.write(f"{gene_a}\t{gene_b}\n")
                            count += 1
                            
            print(f"3. HOÀN TẤT! Đã lưu {count} tương tác vào: {PPI_FILE}")
            
        else:
            print(f"LỖI TỪ SERVER: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"LỖI KẾT NỐI: {e}")

if __name__ == "__main__":
    genes_list = get_gene_list()
    if genes_list:
        fetch_from_string_db(genes_list)