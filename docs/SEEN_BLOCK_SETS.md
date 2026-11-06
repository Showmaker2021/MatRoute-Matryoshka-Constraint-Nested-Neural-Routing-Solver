# Phân bố các tập khối (Block Sets) trong 110 biến thể VRP

Tài liệu này ghi nhận ánh xạ giữa các biến thể bài toán và tập khối ràng buộc kích hoạt $S \subseteq \{\text{DEM}, \text{TW}, \text{PRZ}, \text{PD}, \text{RTE}\}$ theo quy định tại mục 3.3 của `MATROUTE_IMPLEMENTATION_GUIDE.md`.

---

## 1. 11 bài toán huấn luyện (Seen Problems)

Chỉ có **5 tập khối khác nhau** xuất hiện trong quá trình huấn luyện:

| Tập khối $S$ | Danh sách bài toán Seen | Ghi chú |
|---|---|---|
| $\emptyset$ (CORE-only) | `tsp`, `atsp` | Chỉ chạy khối hình học CORE, không có ràng buộc bổ sung. |
| `{"PRZ"}` | `op`, `pctsp` | Bài toán thu thập điểm thưởng / phạt. |
| `{"PD"}` | `pdtsp` | Bài toán lấy/giao hàng có ma trận quan hệ cặp $R$. |
| `{"DEM", "RTE"}` | `cvrp`, `acvrp`, `cvrpb`, `ocvrp` | Biến thể sức chứa cơ bản, có tuyến con / tuyến mở. |
| `{"DEM", "TW", "RTE"}` | `cvrptw`, `ocvrptw` | Biến thể sức chứa kết hợp cửa sổ thời gian. |

---

## 2. Phân bố trên toàn bộ 110 biến thể (11 Seen + 99 Unseen)

Toàn bộ 110 bài toán phân bố vào **6 tập khối**:

| Tập khối $S$ | Số bài Seen | Số bài Unseen | Tổng số bài | Ví dụ các biến thể |
|---|:---:|:---:|:---:|---|
| $\emptyset$ (CORE-only) | 2 | 0 | 2 | `tsp`, `atsp` |
| `{"PRZ"}` | 2 | 4 | 6 | `op`, `pctsp`, `aop`, `spctsp`, `apctsp`, `aspctsp` |
| `{"PD"}` | 1 | 1 | 2 | `pdtsp`, `apdtsp` |
| `{"DEM", "RTE"}` | 4 | 44 | 48 | `cvrp`, `cvrpb`, `ocvrp`, `acvrp`, `mdcvrp`, `amdcvrp`,... |
| `{"DEM", "TW", "RTE"}` | 2 | 46 | 48 | `cvrptw`, `ocvrptw`, `cvrpbtw`, `mdcvrptw`, `amdcvrptw`,... |
| `{"DEM", "PD", "RTE"}` | 0 | 4 | 4 | `pdcvrp`, `apdcvrp`, `opdcvrp`, `aopdcvrp` |
| **Tổng cộng** | **11** | **99** | **110** | |

### Nhận xét quan trọng về cấu trúc tổ hợp:
1. Hầu hết các bài unseen (95/99 bài) vẫn thuộc 5 tập khối đã thấy lúc huấn luyện; sự khác biệt giữa chúng nằm ở các thuộc tính Multi-Depot (`md`), Duration Limit (`l`), Backhaul Priority (`bp`) và tính bất đối xứng (`a`), vốn được giải quyết qua Feasibility Mask của môi trường và ma trận khoảng cách $D, D^T$.
2. Duy nhất 4 bài (`pdcvrp`, `apdcvrp`, `opdcvrp`, `aopdcvrp`) kích hoạt một tập khối hoàn toàn mới là `{"DEM", "PD", "RTE"}` (kết hợp đồng thời nhu cầu hàng hóa và quan hệ cặp pickup-delivery).
3. Do đó, thí nghiệm **Split tổ hợp có kiểm soát P3 (E3)** là bắt buộc để chứng minh năng lực ghép module của MatRoute vượt trội so với Flat baseline.
