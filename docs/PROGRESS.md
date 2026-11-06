# Project Progress & Verification Log (PROGRESS.md)

## Bảng tổng hợp các bước triển khai

| Giai đoạn | Nội dung | Trạng thái | Ghi chú & Nghiệm thu |
|---|---|---|---|
| **0 & 1** | Khảo sát giao diện URS & Tái lập baseline | **HOÀN THÀNH** | Đã tạo `docs/URS_INTERFACE.md`, chạy `scripts/smoke_urs_env.py` qua 11/11 bài seen. Môi trường local ghi nhận tại `docs/ENV_LOCAL.md` (RTX 3050 6GB Laptop GPU, PyTorch 2.7.1+cu118). |
| **2** | Cấu trúc khối (`layout.py`, `spec.py`, `features.py`) | **HOÀN THÀNH** | Pass toàn bộ `test_spec.py` và `test_features.py` (3/3 & 3/3 tests). Sai số feature so với URS `Model_LIB.py` < 1e-6. Phân tích 6 tập khối tại `docs/SEEN_BLOCK_SETS.md`. |
| **3** | Mạng nơ-ron Block-Causal (`matroute/nn/`, `encoder.py`) | **HOÀN THÀNH** | Pass toàn bộ 5 unit test bắt buộc (`tests/test_encoder.py`): prefix equivalence, zero-grad to CORE, mask vs slice, symmetric D/D^T parity, norm independence. |
| **4** | Decoder Funnel & Policy (`decoder.py`, `policy.py`) | **HOÀN THÀNH** | Pass toàn bộ 4 unit test bắt buộc (`tests/test_decoder.py`): funnel=full khi k=N, tính khả thi trên 11 bài seen, CORE-only cho TSP, POMO start node parity với URS. |
| **5** | Trainer & Smoke Training (`matroute/train/`, `train_matroute.py`) | **HOÀN THÀNH** | `tests/test_trainer_overfit.py` pass 2/2: Overfit test TSP-20 giảm cost 30.20% (vượt ngưỡng 15%), checkpoint resume khôi phục 100% loss. CLI `train_matroute.py` chạy trơn tru trên CUDA. |
| **6** | Tester & Đo đạc nhất quán (`matroute/eval/`, `test_matroute.py`) | **HOÀN THÀNH** | Triển khai `matroute/eval/tester.py`, CLI `test_matroute.py`. Pass test `tests/test_tester.py` (3/3): CSV chuẩn định dạng, công thức gap min/max, harness consistency check với URS. Smoke test 64 instances trên 4 bài seen (TSP, CVRP, CVRPTW, PDTSP) chạy trong 14.5s với x8 augmentation ra file `results/e1_smoke_4seen.csv`. |
| **7** | Encoder thưa & Vi kiểm tra tốc độ (`bench_funnel.py`) | **HOÀN THÀNH** | Đã tạo `scripts/bench_funnel.py` đo bước decoder và forward pass encoder trên $N \in \{200, 500, 1000, 2000\}$. Xử lý OOM an toàn, xuất kết quả tại `results/bench_funnel.csv`. Unit test `test_sparse_knn_parity` ($K \ge N$ khớp 100% bản dày, diff < 1e-5) pass. |
| **8** | Gói chạy Server (`SERVER_RUNBOOK.md`, scripts, GATES.md) | **HOÀN THÀNH** | Đã tạo `docs/SERVER_RUNBOOK.md`, `docs/GATES.md` (chi tiết hóa tiêu chí G1, G2, G3 và fallback), cùng 10 kịch bản thực thi hoàn chỉnh (`run_pilot`, `run_full_d128`, `run_full_d192`, `run_eval_all`, `run_ablation` cả `.sh` và `.bat`). Sẵn sàng 100% để chạy huấn luyện trên Server GPU lớn. |
