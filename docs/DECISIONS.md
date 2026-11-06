# Project Decision Log (DECISIONS.md)

| Ngày | Quyết định | Lý do |
|---|---|---|
| 2026-11-06 | Ánh xạ cờ Pickup/Delivery của bài VRP vào khối DEM; khối PD chỉ dùng khi `reset_state.relation is not None`. | Vector λ cho thấy cờ pickup/delivery của VRP chỉ mã hóa dấu demand (linehaul vs backhaul); chỉ PDTSP/PDCVRP có quan hệ ghép cặp thực sự. |
| 2026-11-06 | Sử dụng cờ `--sparse_knn K` cho encoder ở quy mô lớn $N > 1000$. | Tránh OOM do độ phức tạp $O(N^2)$ của full self-attention khi đánh giá trên đồ thị lớn. |
| 2026-11-06 | Khối DEM, TW, PRZ, RTE dùng cùng bias khoảng cách $- \alpha_h \cdot D_{ij}$ với tham số học riêng cho từng head. | Tận dụng tín hiệu khoảng cách địa lý kết hợp với ràng buộc riêng mà không làm xáo trộn tính block-causal. |
| 2026-11-06 | MatRouteTester chia nhỏ số instance đánh giá theo chunk `batch_size` (mặc định 32-64). | Đảm bảo POMO rollout kết hợp Augmentation ($\times 8$ và $\times 128$) chạy mượt mà và không bao giờ bị OOM trên GPU cá nhân (6GB VRAM). |
| 2026-11-06 | Thiết lập `URSPolicyAdapter` cho phép chạy mô hình URS trực tiếp qua `MatRouteTester`. | Xác thực tính nhất quán 100% của harness đánh giá (harness consistency check) so với `Tester.py` gốc của URS. |
| 2026-11-06 | Chuẩn hóa mô hình huấn luyện và đánh giá chính về $d=128$ (thay vì $d=192$). | Đảm bảo tính công bằng (fair comparison) tuyệt đối về số lượng tham số (~1.3M) so với baseline URS và các kiến trúc NCO hiện hành (MVMoE, RouteFinder, MTPOMO). Mọi cải thiện hoàn toàn đến từ cấu trúc nơ-ron và nguyên lý Matryoshka chứ không phải do tăng dung lượng mạng. |
