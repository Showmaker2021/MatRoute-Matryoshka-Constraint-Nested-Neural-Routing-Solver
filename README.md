# MatRoute: Matryoshka Constraint-Nested Neural Routing Solver

> **Xây dựng trên nền tảng môi trường định tuyến hợp nhất URS (ICML 2026)**  
> Tác giả ý tưởng & Thiết kế: Long (@Long)  
> Kiến trúc: **Matryoshka Constraint-Nested Representation Learning + Block-Causal Neural Architecture + Funnel Decoder**  
> Quy chuẩn so sánh: **Chuẩn hóa embedding dimension $d=128$ (~1.3M tham số) để đảm bảo đối sánh công bằng tuyệt đối (Fair Comparison) với URS, MVMoE, RouteFinder và MTPOMO.**

---

## 🌟 Tổng quan về MatRoute

MatRoute là bộ giải định tuyến nơ-ron đa bài toán giải quyết **110 biến thể VRP** bằng nguyên lý học biểu diễn lồng nhau (Matryoshka Representation Learning - MRL).

Thay vì ép mọi đặc trưng vào một không gian nhúng phẳng gây nhiễu chéo (interference), MatRoute chia vector nhúng thành các khối chiều lồng nhau theo thứ tự nhân quả (**Block-Causal Structure**):
- **Khối lõi hình học (`CORE`, 48 chiều)**: Chỉ đọc, bảo vệ tuyệt đối bài toán cơ sở TSP/ATSP khỏi nhiễu gradient của các ràng buộc.
- **Các khối ràng buộc chuyên biệt (80 chiều)**: `DEM` (24d), `TW` (24d), `PRZ` (8d), `PD` (8d), `RTE` (16d).
- **Decoder Funnel (Coarse-to-Fine Retrieval)**: Tầng 0 chấm điểm nhanh bằng chiều thấp của từng khối để lọc ra shortlist ứng viên top-$k$; Tầng 1 rerank bằng toàn bộ chiều vector, giúp tăng tốc vượt bậc ở quy mô lớn $N \ge 1000$.

Toàn bộ mô hình tuân thủ chặt chẽ số chiều $d=128$ nhằm đảm bảo mọi sự vượt trội về kết quả (zero-shot transfer, gap, tốc độ) hoàn toàn đến từ đột phá về mặt cấu trúc biểu diễn lồng nhau, không phụ thuộc vào việc tăng tham số.

---

## ⚙️ 1. Cài đặt Môi trường (Linux & Windows)

### 1.1 Tạo môi trường Conda (`ec_nco`)
Khuyến nghị sử dụng Python 3.10 trên cả Linux (Ubuntu 20.04/22.04 LTS) và Windows (PowerShell/CMD):

```bash
# 1. Tạo và kích hoạt môi trường conda
conda create -n ec_nco python=3.10 -y
conda activate ec_nco

# 2. Cài đặt PyTorch với hỗ trợ CUDA (chọn bản phù hợp với GPU Server)
# Với CUDA 11.8:
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Hoặc với CUDA 12.1:
# pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Cài đặt các thư viện phụ thuộc & công cụ theo dõi Weights & Biases (wandb)
pip install pyyaml pytest numpy scipy matplotlib wandb
```

### 1.2 Kiểm tra môi trường trước khi chạy
Chạy toàn bộ 21 unit tests để đảm bảo autograd, block-causal, mask và môi trường hoạt động chuẩn xác:
```bash
python -m pytest tests/
```
*(Yêu cầu: Toàn bộ 21 tests đều báo `PASSED` trong ~25–35 giây).*

---

## 🚀 2. Hướng dẫn Huấn luyện Mô hình Chuẩn ($d=128$)

> [!NOTE]
> Quy mô huấn luyện đầy đủ: **500 epochs $\times$ 2000 batches $\times$ 128 batch size ở $N=100$**.  
> Thời gian ước tính: ~90–95 giờ trên 1 GPU RTX 4090 hoặc A100.

### 2.1 Chạy Huấn luyện Đầy đủ (Full Training)

#### Trên Linux:
```bash
# Cách 1: Chạy qua kịch bản tự động (tự động phát hiện checkpoint để resume)
bash scripts/run_full_d128.sh

# Cách 2: Chạy trực tiếp qua lệnh CLI (kèm theo dõi W&B)
python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --checkpoint_dir checkpoints/full_d128 \
    --log_dir results/full_d128 \
    --use_wandb \
    --wandb_project MatRoute \
    --wandb_name matroute_d128_full
```

#### Trên Windows (PowerShell / CMD):
```powershell
# Cách 1: Chạy qua file batch
.\scripts\run_full_d128.bat

# Cách 2: Chạy trực tiếp qua lệnh CLI
python train_matroute.py `
    --config matroute/configs/full_d128.yaml `
    --checkpoint_dir checkpoints/full_d128 `
    --log_dir results/full_d128 `
    --use_wandb `
    --wandb_project MatRoute `
    --wandb_name matroute_d128_full
```

---

### 2.2 Huấn luyện Thí điểm (Pilot Run - 50 Epochs)
Để kiểm tra tính hội tụ ban đầu và đối soát **Cổng chất lượng G1** ([`docs/GATES.md`](docs/GATES.md)) trước khi chạy dài 500 epochs:

```bash
# Linux:
bash scripts/run_pilot.sh

# Windows:
.\scripts\run_pilot.bat
```

---

### 2.3 Cơ chế Tự động Phục hồi Huấn luyện (`--resume`)
Nếu máy chủ bị ngắt điện, quá thời gian thuê GPU hoặc tiến trình bị dừng, bạn chỉ cần gõ lại chính kịch bản `run_full_d128` hoặc thêm cờ `--resume`:
```bash
python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --checkpoint_dir checkpoints/full_d128 \
    --resume checkpoints/full_d128/checkpoint_latest.pt \
    --use_wandb
```
Hệ thống sẽ tự động khôi phục toàn bộ: Trọng số mô hình, trạng thái AdamW, Learning Rate Scheduler, thống kê EMA và epoch hiện tại để tiếp tục huấn luyện mà không mất mát dữ liệu.

---

## 📊 3. Hướng dẫn Đánh giá Toàn diện 110 Biến thể (Full Benchmark)

Đánh giá mô hình $d=128$ đã huấn luyện trên toàn bộ **110 bài toán** với Augmentation ($\times 8$ đối xứng, $\times 128$ bất đối xứng) và POMO.

### 3.1 Đánh giá trọn gói 1 lệnh (All-in-One)
Tự động chạy toàn bộ thí nghiệm E1 (11 Seen), E2a (44 Unseen đối xứng), E2b (55 Unseen bất đối xứng) và E6 (Funnel Sweep):

```bash
# Linux:
bash scripts/run_eval_all.sh checkpoints/full_d128/checkpoint_best.pt

# Windows:
.\scripts\run_eval_all.bat checkpoints\full_d128\checkpoint_best.pt
```

---

### 3.2 Đánh giá chi tiết từng nhóm bài toán (CLI Runner)

1. **Thí nghiệm E1: 11 bài toán Seen**
   ```bash
   python test_matroute.py \
       --group seen \
       --model_load checkpoints/full_d128/checkpoint_best.pt \
       --output_csv results/e1_seen.csv
   ```

2. **Thí nghiệm E2a: 44 bài toán Unseen đối xứng (Bảng 11 của URS)**
   ```bash
   python test_matroute.py \
       --group unseen44 \
       --model_load checkpoints/full_d128/checkpoint_best.pt \
       --output_csv results/e2_unseen44.csv
   ```

3. **Thí nghiệm E2b: 55 bài toán Unseen bất đối xứng & PDCVRP (Bảng 12 của URS)**
   ```bash
   python test_matroute.py \
       --group unseen55 \
       --model_load checkpoints/full_d128/checkpoint_best.pt \
       --output_csv results/e2_unseen55.csv
   ```

4. **Thí nghiệm E6: Kiểm tra tốc độ & chất lượng Decoder Funnel**
   ```bash
   python test_matroute.py \
       --problems cvrp,cvrptw \
       --level FUNNEL \
       --model_load checkpoints/full_d128/checkpoint_best.pt \
       --output_csv results/e6_funnel.csv
   ```

Các file CSV kết quả sinh ra tại thư mục `results/` có đầy đủ các cột chuẩn bài báo:  
`variant, group, gap, no_aug_gap, time, n, model, seed`

---

## 🔬 4. Nghiên cứu Bóc tách (Ablation Studies - E8)

Chạy các biến thể thực nghiệm trên cùng ngân sách $d=128$:
- **`--soft_core`**: Kiểm tra cơ chế cổng mềm $\tanh(g)$ so với lõi chỉ-đọc (Read-only).
- **`--stopgrad_core`**: Ngắt gradient ngược về CORE.

```bash
# Linux:
bash scripts/run_ablation.sh

# Windows:
.\scripts\run_ablation.bat
```

---

## 📈 5. Hướng dẫn Theo dõi Trực quan bằng Weights & Biases (W&B)

MatRoute tích hợp sẵn module **Weights & Biases (`wandb`)** giúp toàn bộ nhóm nghiên cứu theo dõi trực tiếp các đường cong học tập, độ dài tour theo từng bài toán và phát hiện sớm hiện tượng phân kỳ từ xa qua điện thoại hoặc trình duyệt web.

### 5.1 Bước 1: Đăng nhập tài khoản W&B (Chỉ cần làm 1 lần trên Server)
1. Đăng ký tài khoản miễn phí tại [wandb.ai](https://wandb.ai) (nếu chưa có).
2. Lấy API Key cá nhân tại [wandb.ai/authorize](https://wandb.ai/authorize).
3. Mở terminal trên server và chạy lệnh đăng nhập:
```bash
wandb login
# Dán API Key của bạn vào terminal và nhấn Enter
```

---

### 5.2 Bước 2: Kích hoạt W&B khi Huấn luyện

#### Trên Linux:
```bash
python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --checkpoint_dir checkpoints/full_d128 \
    --log_dir results/full_d128 \
    --use_wandb \
    --wandb_project MatRoute \
    --wandb_name matroute_d128_full
```

#### Trên Windows (PowerShell):
```powershell
python train_matroute.py `
    --config matroute/configs/full_d128.yaml `
    --checkpoint_dir checkpoints/full_d128 `
    --log_dir results/full_d128 `
    --use_wandb `
    --wandb_project MatRoute `
    --wandb_name matroute_d128_full
```

#### Bổ sung: Huấn luyện nhóm (Team/Organization)
Nếu nhóm bạn có W&B Team / Entity chung để cả lab cùng vào xem:
```bash
python train_matroute.py \
    --config matroute/configs/full_d128.yaml \
    --checkpoint_dir checkpoints/full_d128 \
    --use_wandb \
    --wandb_project MatRoute \
    --wandb_entity <ten_team_cua_ban> \
    --wandb_name matroute_d128_run1
```

---

### 5.3 Chế độ Ngoại tuyến (Offline Mode - Khi Server không có Internet trực tiếp)
Nếu GPU Server trong mạng nội bộ không thể kết nối trực tiếp ra ngoài Internet:
1. Chạy với cờ `--wandb_mode offline`:
   ```bash
   python train_matroute.py \
       --config matroute/configs/full_d128.yaml \
       --checkpoint_dir checkpoints/full_d128 \
       --use_wandb \
       --wandb_mode offline
   ```
2. Sau khi chạy xong hoặc khi máy chủ có kết nối mạng, đồng bộ dữ liệu lên web bằng lệnh:
   ```bash
   wandb sync wandb/offline-run-*
   ```

---

### 5.4 Các Biểu đồ và Chỉ số Tự động Hiển thị trên Dashboard W&B
Khi W&B được kích hoạt, trang Dashboard sẽ tự động vẽ theo thời gian thực (real-time):
- **`train/loss_total`**: Tổng hàm mất mát kết hợp.
- **`train/loss_rl`**: Mất mát REINFORCE với POMO baseline.
- **`train/loss_dist`**: Mất mát Kullback–Leibler (KL) chưng cất tri thức từ chính sách FULL xuống FUNNEL và CORE.
- **`train/cost_metric`**: Chi phí / độ dài tour trung bình của batch.
- **`tasks/<variant>/cost`**: Biểu đồ phân rã chi phí riêng cho từng biến thể bài toán (ví dụ: `tasks/tsp/cost`, `tasks/cvrp/cost`, `tasks/cvrptw/cost`...), giúp theo dõi bài toán nào đang tiến bộ và bài nào gặp trở ngại.
- **`train/lambda_dist`**: Hệ số điều biến chưng cất suy giảm tuyến tính từ $0.10 \to 0.02$.
- **`train/lr`**: Tỷ lệ học của thuật toán tối ưu AdamW.

---

## 📁 6. Cấu trúc Thư mục Dự án

```text
URS/
├── matroute/                       # Package mã nguồn MatRoute
│   ├── layout.py                   # Cấu hình BlockLayout (d128 mặc định, d64, d192)
│   ├── spec.py                     # Bản đồ ánh xạ 110 biến thể sang khối S
│   ├── features.py                 # Trích xuất đặc trưng khớp số học với URS (< 1e-6)
│   ├── nn/                         # Các tầng nơ-ron Block-Causal
│   │   ├── block_linear.py         # Tuyến tính phân khối có mặt nạ nhân quả
│   │   ├── block_norm.py           # InstanceNorm độc lập từng khối
│   │   ├── block_attn.py           # Block Attention (D/D^T heads, PD bias, kNN)
│   │   └── block_ffn.py            # FFN nhân quả
│   ├── encoder.py                  # Encoder lồng nhau (L=12 layers)
│   ├── decoder.py                  # MatRoute Decoder & funnel_logits
│   ├── policy.py                   # MatRoutePolicy POMO rollout & distillation
│   ├── train/                      # Trainer, SandwichSampler, EMA Stats
│   └── eval/                       # Tester, Groups phân hoạch 110 bài toán
├── train_matroute.py               # Điểm kích hoạt huấn luyện (CLI)
├── test_matroute.py                # Điểm kích hoạt đánh giá (CLI)
├── scripts/                        # Kịch bản thực thi tự động (.sh & .bat)
│   ├── bench_funnel.py             # Vi kiểm tra tốc độ Funnel & Encoder
│   ├── run_pilot.*                 # Kịch bản huấn luyện Pilot 50 epochs
│   ├── run_full_d128.*             # Kịch bản huấn luyện Full d=128
│   ├── run_eval_all.*              # Kịch bản đánh giá toàn diện 110 bài toán
│   └── run_ablation.*              # Kịch bản bóc tách
├── docs/                           # Tài liệu thiết kế & vận hành
│   ├── URS_INTERFACE.md            # Báo cáo khảo sát giao diện URS
│   ├── SERVER_RUNBOOK.md           # Cẩm nang chi tiết vận hành Server GPU
│   ├── GATES.md                    # Tiêu chí đối soát Cổng chất lượng G1, G2, G3
│   ├── PROGRESS.md                 # Nhật ký tiến độ chi tiết
│   └── DECISIONS.md                # Các quyết định thiết kế kỹ thuật
└── tests/                          # Toàn bộ 21 Unit Tests tự động
```

---

## 📜 7. Môi trường Gốc URS (ICML 2026 Background)

Các file nguyên bản của tác giả URS (`env/`, `problem/`, `data/`, `model/`, `train.py`, `test.py`, `Trainer.py`, `Tester.py`) được giữ nguyên vẹn 100% để phục vụ việc trích xuất môi trường, tập dữ liệu chuẩn và làm đối chứng đo lường chính xác nhất.
