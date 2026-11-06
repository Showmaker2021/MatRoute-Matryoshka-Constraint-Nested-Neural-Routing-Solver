# Server Execution Runbook (SERVER_RUNBOOK.md)

Tài liệu hướng dẫn triển khai, huấn luyện và đánh giá mô hình **MatRoute (Matryoshka Constraint-Nested Neural Routing Solver)** trên hệ thống GPU Server (Linux/Windows, khuyến nghị RTX 4090, A100 hoặc H100).

---

## 1. Môi trường & Thiết lập ban đầu

### 1.1 Yêu cầu hệ thống
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04 LTS) hoặc Windows Server
- **GPU**: Tối thiểu 1x GPU NVIDIA $\ge 16\text{GB}$ VRAM (khuyến nghị RTX 4090 24GB hoặc A100 40/80GB)
- **CUDA**: 11.8 hoặc 12.x
- **Dung lượng đĩa**: Tối thiểu 50GB trống (dataset: ~5GB, checkpoints & logs: ~20GB)

### 1.2 Dựng môi trường Conda (`ec_nco`)
```bash
# Tạo môi trường Python 3.10
conda create -n ec_nco python=3.10 -y
conda activate ec_nco

# Cài đặt PyTorch với CUDA 11.8 (hoặc phù hợp với driver server)
pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện phụ thuộc
pip install pyyaml pytest numpy scipy matplotlib
```

### 1.3 Cấu trúc dữ liệu trên Server
Đảm bảo thư mục dữ liệu đã tải đầy đủ 110 bài toán đặt tại:
```text
URS/
├── dataset/            # Chứa 110 thư mục bài toán (tsp/, cvrp/, cvrptw/...)
├── checkpoints/        # Nơi lưu checkpoint tự động
├── results/            # Nơi lưu logs và CSV kết quả
├── matroute/           # Source code của MatRoute
├── docs/               # Tài liệu, runbook và gates
└── scripts/            # Kịch bản thực thi (.sh và .bat)
```

---

## 2. Kiểm tra trước chuyến bay (Pre-flight Checks)

Trước khi kích hoạt các lệnh chạy dài, chạy kiểm tra nhanh để đảm bảo môi trường và autograd hoàn toàn bình thường:

```bash
# 1. Chạy toàn bộ 20 unit tests (thời gian: ~25s)
python -m pytest tests/

# 2. Đo đạc sơ bộ tốc độ Funnel Decoder & Encoder
python scripts/bench_funnel.py --sizes 200 500 1000 --preset d128
```
*Tất cả 20 tests phải báo `PASSED` trước khi sang Bước 3.*

---

## 3. Lịch trình thực nghiệm & Ước tính thời gian

Dưới đây là thứ tự thực hiện các pha thực nghiệm theo kế hoạch trong proposal:

| Pha | Kịch bản thực thi | Số Epoch / Instance | Mục tiêu | Thời gian ước tính (RTX 4090) |
|---|---|---|---|---|
| **Pha 1** | `scripts/run_pilot.sh` | 50 epochs | Kiểm tra hội tụ ban đầu, đối soát **Gate G1** | ~8 - 10 giờ |
| **Pha 2** | `scripts/run_full_d128.sh` | 500 epochs | Huấn luyện mô hình $d=128$ (so khớp số tham số với URS) | ~90 - 95 giờ |
| **Pha 3** | `scripts/run_full_d192.sh` | 500 epochs | Huấn luyện mô hình chính $d=192$ (Primary Model) | ~135 - 140 giờ |
| **Pha 4** | `scripts/run_eval_all.sh` | 110 variants | Đánh giá E1 (11 Seen), E2 (99 Unseen), E6 (Funnel) | ~6 - 8 giờ |
| **Pha 5** | `scripts/run_ablation.sh` | 50 epochs x 2 | Nghiên cứu bóc tách E8: `--soft_core`, `--stopgrad_core` | ~16 - 20 giờ |

---

## 4. Hướng dẫn thực thi từng bước

### Bước 1: Huấn luyện Thí điểm (Pilot Run - 50 Epochs)
```bash
bash scripts/run_pilot.sh
# Hoặc trên Windows:
scripts\run_pilot.bat
```
* **Nghiệm thu**: Mở file `results/pilot_eval.csv` và đối chiếu với tiêu chí trong [`docs/GATES.md`](file:///d:/URS/docs/GATES.md). Nếu đạt cổng **G1**, tiếp tục sang Bước 2.

### Bước 2: Huấn luyện Mô hình Đầy đủ ($d=128$)
```bash
bash scripts/run_full_d128.sh
```
* Tự động lưu checkpoint định kỳ vào `checkpoints/full_d128/checkpoint_latest.pt` và bản tốt nhất vào `checkpoint_best.pt`.
* Đánh giá đối soát cổng **G2** (MTL vs STL) tại `results/full_d128/seen_eval.csv`.

### Bước 3: Huấn luyện Mô hình Chính ($d=192$)
```bash
bash scripts/run_full_d192.sh
```
* Đây là mô hình chủ lực cho bài báo, khai thác tối đa cấu trúc Matryoshka với 80 chiều CORE và 112 chiều chia cho các khối ràng buộc.

### Bước 4: Đánh giá Toàn diện 110 Biến thể (E1, E2, E6)
Chạy kịch bản đánh giá với checkpoint tốt nhất:
```bash
bash scripts/run_eval_all.sh checkpoints/full_d192/checkpoint_best.pt
```
Kết quả sẽ xuất ra 4 file CSV sẵn sàng đưa vào bảng biểu bài báo:
- `results/e1_seen.csv`: Bảng kết quả 11 bài toán Seen.
- `results/e2_unseen44.csv`: Bảng kết quả 44 bài toán Unseen đối xứng (Table 11 của URS).
- `results/e2_unseen55.csv`: Bảng kết quả 55 bài toán Unseen bất đối xứng & PDCVRP (Table 12 của URS).
- `results/e6_funnel.csv`: Bảng đối chứng tốc độ và chất lượng của Funnel Decoder.

### Bước 5: Chạy Nghiên cứu Bóc tách (Ablation Study - E8)
```bash
bash scripts/run_ablation.sh
```
* So sánh định lượng giữa mô hình mặc định (Read-only CORE) với `--soft_core` (cổng mềm) và `--stopgrad_core` (ngắt gradient).

---

## 5. Cơ chế Phục hồi khi bị gián đoạn (`--resume`)

Các kịch bản huấn luyện đều đã được cấu hình cờ tự động phục hồi. Nếu server bị ngắt điện hoặc quá giờ thuê (preempted), bạn chỉ cần gõ lại chính kịch bản đó:
```bash
bash scripts/run_full_d192.sh
```
Hệ thống sẽ:
1. Phát hiện file `checkpoints/full_d192/checkpoint_latest.pt`.
2. Khôi phục hoàn toàn: Trọng số mô hình, trạng thái Optimizer AdamW, Learning Rate Scheduler, thống kê EMA của `RunningRewardStats`, epoch hiện tại và random seed.
3. Huấn luyện tiếp tục mà không làm mất dữ liệu trước đó.

---

## 6. Xử lý sự cố thường gặp (Troubleshooting)

1. **Lỗi `CUDA out of memory` (OOM) khi đánh giá**:
   - Mặc định kịch bản test sử dụng `batch_size=64`. Nếu kiểm thử ở quy mô lớn $N \ge 1000$, truyền cờ `--batch_size 16` hoặc `--batch_size 8`.
2. **Không tìm thấy dữ liệu test cho bài toán `xyz`**:
   - Kiểm tra đường dẫn thư mục `dataset/xyz/`. Đảm bảo file `.pkl` hoặc `.txt` tồn tại theo định dạng của URS.
3. **Loss không giảm sau 10 epoch đầu**:
   - Đối chiếu với [`docs/GATES.md`](file:///d:/URS/docs/GATES.md) phần Gate G1. Kiểm tra tỷ lệ lấy mẫu trong `SandwichSampler` và thử nghiệm tùy chọn `--soft_core`.
