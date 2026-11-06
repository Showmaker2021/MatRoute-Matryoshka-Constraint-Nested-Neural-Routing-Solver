# Quality Gates & Fallback Protocols (GATES.md)

Tài liệu này chi tiết hóa 3 cổng chất lượng (Gates G1, G2, G3) theo đề xuất của Proposal MatRoute (Mục 9) để chủ dự án đối soát và ra quyết định khi chạy thực nghiệm huấn luyện trên Server.

---

## 1. Cổng G1: Pilot Verification (Sau 50 Epochs Huấn Luyện Thí Điểm)

### Mục đích
Kiểm tra tính hội tụ ban đầu, xác nhận không có hiện tượng xung đột gradient hủy hoại khối `CORE` (geometry interference), và kiểm tra tính ổn định số học (không xuất hiện NaN/Inf loss).

### Nguồn dữ liệu kiểm tra
- Log huấn luyện: `results/pilot/train_log.csv`
- File đánh giá: `results/pilot_eval.csv` (sinh bởi lệnh `test_matroute.py --group seen --max_instances 100`)

### Tiêu chí nghiệm thu G1
| Chỉ số | Điều kiện Đạt | Điều kiện Cảnh báo | Điều kiện Thất bại |
|---|---|---|---|
| **RL Loss hội tụ** | Giảm đơn điệu qua các epoch, không có gai bất thường | Giảm chậm sau epoch 20 | Loss phân kỳ hoặc NaN |
| **Độ dài Tour TSP-100** | Giảm ít nhất **35%** so với khởi tạo ngẫu nhiên | Giảm 20% - 35% | Giảm < 20% |
| **Gap trung bình 11 Seen** | $\le 1.15 \times \text{URS-MTL}$ cùng số bước | $1.15 \times - 1.30 \times$ | $> 1.30 \times$ |
| **Distillation Loss** | Giảm dần theo lịch $\lambda_{\text{dist}}$ (từ 0.1 $\to$ 0.02) | Dao động | Phân kỳ |

### Giao thức xử lý khi KHÔNG đạt G1 (Fallback)
1. **Nếu loss phân kỳ hoặc có NaN**:
   - Kiểm tra logit clipping scale $\zeta=50.0$, giảm learning rate từ `1e-4` xuống `5e-5`.
2. **Nếu khối CORE học quá chậm do bị cô lập**:
   - Kích hoạt tùy chọn `--soft_core` (cho phép 4 lớp trên cùng của Encoder đọc mềm thông qua cổng $\tanh(g)$).
3. **Nếu chưng cất kéo lệch chính sách FULL**:
   - Giảm hệ số khởi đầu $\lambda_{\text{dist}}$ từ 0.1 xuống 0.05.

---

## 2. Cổng G2: MTL Interference & STL Parity (Sau khi hoàn tất Full Training d=128/d=192)

### Mục đích
Kiểm chứng giả thuyết **H2**: Lõi hình học chỉ-đọc (read-only core) có bảo vệ được bài toán cơ sở và thu hẹp khoảng cách giữa huấn luyện đa bài toán (MTL) so với đơn bài toán (STL) hay không.

### Nguồn dữ liệu kiểm tra
- File đánh giá Seen: `results/e1_seen.csv`
- So sánh đối chuẩn với bảng 3 & 4 của bài báo URS (`URS-MTL` vs `URS-STL`).

### Tiêu chí nghiệm thu G2
| Phép so sánh | Tiêu chuẩn Đạt |
|---|---|
| **Gap TSP-100** | Khoảng cách suy giảm MTL so với STL giảm $\ge 40\%$ so với mức suy giảm của URS-MTL. |
| **Gap CVRP-100** | Gap đạt mức cạnh tranh với URS-MTL ($\le 1.85\%$). |
| **Cân bằng hiệu năng** | Không có bài toán seen nào bị suy giảm hiệu năng quá $15\%$ so với baseline tương ứng. |

### Giao thức xử lý khi KHÔNG đạt G2
1. Tăng tỷ trọng lấy mẫu của bài toán yếu trong `SandwichSampler` (ưu tiên dynamic task weighting).
2. Tăng số lượng head cho ma trận khoảng cách $D$ và $D^T$ trong khối `CORE`.

---

## 3. Cổng G3: Zero-shot Combinatorial Transfer & Large Scale Funnel (Đánh giá Unseen & Funnel)

### Mục đích
Kiểm chứng giả thuyết **H1** (Nested representation cải thiện zero-shot trên 99 bài biến thể chưa từng thấy) và **H3** (Funnel Decoder giữ chất lượng và tăng tốc rõ rệt ở $N \ge 1000$).

### Nguồn dữ liệu kiểm tra
- Unseen 44 (Symmetric): `results/e2_unseen44.csv`
- Unseen 55 (Asymmetric & PDCVRP): `results/e2_unseen55.csv`
- Large-scale Funnel sweep: `results/e4_large.csv` và `results/e6_funnel.csv`

### Tiêu chí nghiệm thu G3
| Tiêu chí | Ngưỡng chấp nhận |
|---|---|
| **Gap trung bình Unseen 44** | Cải thiện ít nhất **$0.8\%$** gap so với URS-MTL phẳng. |
| **Gap Multi-Depot (MD-CVRP, MD-VRPTW...)** | Cải thiện rõ rệt: giảm gap tối thiểu **$3.0\%$** so với URS-MTL (vốn có gap 15–48%). |
| **Funnel Recall@K** | $\ge 90\%$ node được FULL chọn nằm trong shortlist của Funnel ở các bước ra quyết định. |
| **Chênh lệch Gap Funnel vs Full** | Đánh đổi gap $\le 1.0\%$ nhưng tốc độ giải mã tăng $\ge 2.0\times$ ở $N \ge 1000$. |

### Giao thức xử lý khi KHÔNG đạt G3
1. **Nếu Recall@K của Funnel thấp**:
   - Tăng $k$ của khối `CORE` từ 50 lên 80, và các khối ràng buộc từ 20 lên 32.
2. **Nếu bài toán Unseen có TW hoặc Prize yếu**:
   - Chạy ablation `--flat` hoặc `--core_ladder` để kiểm tra mức độ phụ thuộc biểu diễn.
