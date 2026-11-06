# MatRoute: hướng dẫn implement cho agent (v2)

Tài liệu này dành cho agent triển khai MatRoute trong một fork của repo URS (CIAM-Group/URS). Proposal đầy đủ (motivation, thí nghiệm, rủi ro) nằm ở tài liệu "Proposal: Matryoshka Constraint-Nested Neural Routing Solver"; file này chỉ nói **làm gì, theo thứ tự nào, và cách nghiệm thu**.

**Thay đổi so với v1:** giai đoạn 0 (khảo sát) và 1 (tái lập baseline URS) đã hoàn thành; các giao diện đã xác nhận được ghi ở mục 1. Ánh xạ khối PD/DEM được sửa lại theo vector λ thật. Thêm mục 2 về hai chế độ chạy: máy local chỉ để kiểm tra chức năng, huấn luyện và đánh giá đầy đủ chạy trên server.

## 0. Nguyên tắc làm việc

1. **Không sửa mã gốc của URS** (`env/`, `problem/`, `data/`, `utils/`, `model/`, `Trainer.py`, `Tester*.py`, `train.py`, `test.py`, `args.py`). Chỉ import. Mọi mã mới nằm trong `matroute/` và hai entry point `train_matroute.py`, `test_matroute.py` ở thư mục gốc. Nếu buộc phải sửa file gốc, dừng lại và hỏi.
2. **Không đoán API.** Mọi chỗ ghi `[XÁC NHẬN]` nghĩa là phải đọc mã URS đúng đoạn được chỉ ra và ghi kết quả vào `docs/URS_INTERFACE.md` (mục bổ sung) trước khi viết mã phụ thuộc vào nó.
3. **Mỗi giai đoạn có tiêu chí nghiệm thu.** Không sang giai đoạn sau nếu chưa qua. Nếu một tiêu chí không đạt, báo cáo số liệu thật và dừng, không chỉnh ngưỡng cho khớp.
4. **Không bịa số liệu.** Không điền kết quả vào bảng khi chưa chạy. Số của URS lấy từ paper chỉ dùng làm tham chiếu, ghi rõ nguồn.
5. **Hỏi khi bị chặn.** Môi trường không chạy được, kiểm thử đơn vị không qua sau hai lần sửa, hoặc cần sửa file gốc của URS: dừng và hỏi.
6. Ghi mọi quyết định thiết kế không có trong tài liệu này vào `docs/DECISIONS.md` (một dòng: quyết định, lý do, ngày).
7. **Giấy phép.** URS dùng AGPLv3 (đã xác nhận ở `LICENSE`). Mã dẫn xuất của nó khi phân phối hoặc chạy như dịch vụ mạng phải công khai dưới cùng giấy phép. Giữ nguyên thông báo bản quyền gốc, và thêm `LICENSE` AGPLv3 vào fork. Đây không phải tư vấn pháp lý; nếu chủ dự án muốn giấy phép khác thì phải hỏi trước khi phát hành.

## 1. Những gì đã xác nhận về URS (không cần khảo sát lại)

Nguồn: `docs/URS_INTERFACE.md` do agent khảo sát tạo. Giai đoạn 1 đã được chủ dự án chạy lại với dữ liệu và checkpoint tải từ Google Drive và xác nhận thành công.

**Môi trường `env/UniVRPEnv.py`** (không import gì từ `model/`, nên dùng trực tiếp, không cần lớp bọc):

- `env.load_problems(batch_size, problem_name, problem_size, pomo_size, capacity, device, aug_factor, lib_data/validation_data)`. `pomo_size=None` thì bằng `problem_size`. MD: `depot_num=3`, `pomo_size = N*3`. PD: `pomo_size` bị giới hạn `N//2`. TSP/ATSP: `depot_num=0`; còn lại `depot_num=1`.
- `env.reset()` trả `(reset_state, None, False)`:
  - `reset_state.problems`: `[B, N+depot_num, 9]`, cột `[x, y, demand, prize, fake_prize, penalty, tw_start, tw_end, service_time]`.
  - `reset_state.dist`: `[B, N+depot_num, N+depot_num]`; `reset_state.log_scale`: `log2(N)`; `reset_state.relation`: `[B, N+1, N+1]` cho bài PD, ngược lại `None`.
- `env.step(selected, mask_fn)` với `selected` kiểu long `[B, P]`, `mask_fn = mask_registry.build_combined_mask(problem_name)`. Trả `(step_state, reward, done)`:
  - `step_state`: `current_node`, `load`, `time`, `length`, `tour_maxlength`, `collected_prize`, `ninf_mask`, `finished`, đều `[B, P]` trừ `ninf_mask` là `[B, P, N+depot_num]` với giá trị `0` (khả thi) hoặc `-inf` (cấm). **Feasible = `ninf_mask == 0`.**
  - `reward` chỉ trả ở bước cuối (`done=True`), `None` ở các bước trước. Dấu: âm khoảng cách cho bài VRP/TSP/MD/PCTSP (đã gồm penalty); **dương** tổng prize cho OP/AOP. Vậy "reward lớn hơn là tốt hơn" đúng cho mọi bài.
- Trạng thái động truy cập được qua `step_state` (và thuộc tính của env): tải còn lại `load` (đã chuẩn hóa, reset về 1.0 khi về depot), thời gian hiện tại `time`, `collected_prize`, `tour_maxlength` (độ dài còn lại cho OP), `length`.

**Dữ liệu `problem/ProblemDef.py`:** `get_random_problems(...)` trả dict 10 key: `xy, demand, dist, prize, penalty, fake_prize, service_time, tw_start, tw_end, route_limit`. Bài bất đối xứng có `xy` bằng 0.

**Augmentation:** ×8 (đối xứng) gọi qua `aug_factor=8` trong `load_problems` (hàm `augment_xy_data_by_8_fold`). ×128 (bất đối xứng) nhân bản batch 128 lần trong env; **nhiễu định danh η được sinh trong mô hình URS** (`Model_LIB.py` L212-220, `node_random_emb = torch.rand(...)`), không phải trong env. MatRoute phải tự sinh η mới cho từng bản sao augmentation.

**Vector λ 13 chiều** (`ProblemRep.py`): `[identifier, coordinates, demand, prize, penalty, early, late, service, depot, pickup, delivery, multi_route, open_route]`. Giá trị của 11 bài seen:

| Bài | λ |
| --- | --- |
| tsp | `[0,1,0,0,0,0,0,0,0,0,0,0,0]` |
| atsp | `[1,0,0,0,0,0,0,0,0,0,0,0,0]` |
| cvrp | `[0,1,1,0,0,0,0,0,1,0,1,1,0]` |
| acvrp | `[1,0,1,0,0,0,0,0,1,0,1,1,0]` |
| cvrpb | `[0,1,1,0,0,0,0,0,1,1,1,1,0]` |
| cvrptw | `[0,1,1,0,0,1,1,1,1,0,1,1,0]` |
| ocvrp | `[0,1,1,0,0,0,0,0,1,0,1,1,1]` |
| ocvrptw | `[0,1,1,0,0,1,1,1,1,0,1,1,1]` |
| op | `[0,1,0,1,0,0,0,0,1,0,0,0,0]` |
| pctsp | `[0,1,0,1,1,0,0,0,1,0,0,0,0]` |
| pdtsp | `[0,1,0,0,0,0,0,0,1,1,1,0,0]` |

**Quy ước tên và danh sách bài:** tên biến thể chữ thường dạng `[a][md][o]cvrp[b|bp][l][tw]`. `ProblemSet.get(name="train_problem_list")` (11 bài seen), `"zero_shot_list"` (99 bài), `"symmetric_list"`, `"asymmetric_list"`, `"all_evaluated_list"` (110). Lọc theo tiền tố bằng `ProblemSet.get(included=..., excluded=...)`.

**Huấn luyện URS (`Trainer.py`):** mỗi batch `random.choice(train_problem_list)`, env và mask khởi tạo lại cho từng batch, `pomo_size = problem_size`, baseline POMO là trung bình reward theo trajectory. Checkpoint tốt nhất `best_checkpoint.pt`, định kỳ `checkpoint_{epoch}.pt`.

**Đánh giá:** gap tính trong `Tester.py` L283-288 (cực đại hóa cho OP). Oracle theo `OracleDict` trong `DataFinder.py`. Dữ liệu test và checkpoint đã được chủ dự án tải về từ Drive (đường dẫn do chủ dự án cung cấp; đặt trong file cấu hình, không hard-code).

**Các điểm còn lại cần xác nhận khi viết mã** (đọc đúng đoạn, ghi vào `docs/URS_INTERFACE.md` mục "Bổ sung"):

- `model/Model.py` L101-135: logic chọn node xuất phát của POMO cho từng nhóm bài (TSP/ATSP, VRP, MD, PD). MatRoute phải tái tạo y hệt.
- `model/Model_LIB.py` L200-307: cách dựng đặc trưng node (cờ depot, pickup, delivery, sub-route, open; chuẩn hóa demand; chuẩn hóa dist; ma trận quan hệ R; xử lý SPCTSP với `fake_prize`). Thứ tự node trong tensor (depot ở đầu?) và chỉ số các depot khi MD.
- `Trainer.py` L237-314: giá trị `capacity` theo từng bài, kích thước batch, cách tạo env mỗi batch.
- `Tester.py` L102-103, L166, L274-304: cách chọn nghiệm tốt nhất qua trajectory và augmentation, định dạng file kết quả.
- `UniVRPEnv.py` L285-286: URS xác định open-route bằng `'o' in problem_name`. MatRoute **không** parse tên mà dùng λ[12] và λ[11] từ `ProblemRep`.

## 2. Hai chế độ chạy: local và server

Mã được viết và kiểm tra trên máy local của chủ dự án (Windows, repo ở `D:\URS` theo báo cáo khảo sát). Huấn luyện và đánh giá đầy đủ chạy trên server. Vì vậy từ giai đoạn 5 trở đi agent **không chạy huấn luyện đầy đủ**; nhiệm vụ là (a) bảo đảm mã đúng bằng các lần chạy nhỏ, (b) chuẩn bị gói chạy server.

**Quy tắc cho máy local:**

- Mọi lệnh có cờ giới hạn: `--max_steps`, `--max_instances`, `--max_epochs`. Mỗi lệnh phải xong trong khoảng 10 phút; nếu lâu hơn, giảm kích thước.
- Windows: dùng `pathlib`, không hard-code đường dẫn kiểu POSIX; `num_workers=0`; không dùng `torch.compile`; `--device auto` (cuda nếu có, ngược lại cpu); bf16 chỉ bật khi GPU hỗ trợ, mặc định fp32 (`--amp none|bf16`).
- Đầu tiên chạy `scripts/env_info.py` ghi `docs/ENV_LOCAL.md`: phiên bản Python, PyTorch, CUDA, tên GPU, bộ nhớ GPU, RAM.
- Mọi seed cố định, ghi seed vào log.

**Các bộ cấu hình:**

| Preset | Dùng ở đâu | Nội dung |
| --- | --- | --- |
| `smoke_local.yaml` | Local | N=20, 2–4 bài (TSP, CVRP, CVRPTW, PDTSP), batch 16–32, ≤ 300 bước, d=64 mô hình nhỏ để chạy nhanh |
| `micro_local.yaml` | Local | Như smoke nhưng N=50 và d=128, ≤ 1,000 bước, dùng để xem đường cong loss |
| `pilot_server.yaml` | Server | Bản d=128, 4 bài, khoảng 100 epoch (pilot cổng G1) |
| `full_d128.yaml`, `full_d192.yaml` | Server | Lịch đầy đủ 500 × 2000 × 128 của URS |
| `ablation/*.yaml` | Server | Lịch rút gọn 250 epoch cho ablation |

Gói chạy server (giai đoạn 8) là sản phẩm của agent nhưng **chủ dự án mới là người chạy**.

## 3. Giai đoạn 2: cấu trúc khối (`matroute/layout.py`, `spec.py`, `features.py`)

### 3.1 Cấu hình khối

| Khối | d=192 | d=128 | d=64 (smoke) |
| --- | --- | --- | --- |
| CORE | 80 | 48 | 24 |
| DEM | 32 | 24 | 8 |
| TW | 32 | 24 | 8 |
| PRZ | 16 | 8 | 8 |
| PD | 16 | 8 | 8 |
| RTE | 16 | 16 | 8 |

Kích thước khối là bội của head-dim 8 (cột d=64 chỉ dùng cho smoke test chức năng, không dùng để báo cáo kết quả).

```python
@dataclass(frozen=True)
class BlockLayout:
    names: tuple            # ("CORE","DEM","TW","PRZ","PD","RTE")
    sizes: dict             # {"CORE":80, ...}
    head_dim: int = 8
    def dim(self): ...
    def idx(self, name) -> LongTensor
    def n_heads(self, name) -> int
    def active_idx(self, S) -> LongTensor   # CORE + các khối trong S, thứ tự cố định
```

### 3.2 Ánh xạ đặc trưng sang khối (`features.py`)

Khác với v1: vector λ cho thấy cờ `pickup`/`delivery` trong bài VRP chỉ mã hóa **dấu của demand** (CVRP có `delivery=1` cho mọi khách; CVRPB có thêm `pickup=1` cho backhaul), nên chúng thuộc khối DEM, không thuộc PD. Khối PD chỉ dành cho **bài ghép cặp pickup–delivery thật sự**, nhận biết bằng `reset_state.relation is not None`.

| Khối | Đặc trưng đầu vào | Trạng thái động (vào truy vấn decoder) |
| --- | --- | --- |
| CORE | `x, y` (cột 0–1 của `problems`), η, cờ depot | không có |
| DEM | `demand` (cột 2), cờ delivery (demand > 0), cờ pickup/backhaul (demand < 0) | `load` |
| TW | `tw_start, tw_end, service_time` (cột 6, 7, 8) | `time` |
| PRZ | `prize` (cột 3; với SPCTSP dùng `fake_prize` cột 4 nếu `Model_LIB` làm vậy `[XÁC NHẬN]`), `penalty` (cột 5) | `collected_prize`, `tour_maxlength` |
| PD | cờ pickup (nửa đầu khách), cờ delivery (nửa sau) theo quy ước N/2, ma trận quan hệ `relation` đưa vào bias attention | không có |
| RTE | cờ multi_route (λ[11]), cờ open_route (λ[12]), phát rộng thành đặc trưng hằng số cho mọi node | không có |

η: bằng 0 cho bài đối xứng; cho bài bất đối xứng (λ[0] = 1) lấy `U(0,1)` mới cho từng bản sao augmentation, theo `Model_LIB.py` L212-220. Cờ depot lấy theo λ[8] và vị trí node (depot ở đầu; MD có 3 depot đầu `[XÁC NHẬN]`).

Mỗi khối có một `nn.Linear` riêng từ đặc trưng của nó vào chiều của khối. `build_block_inputs(reset_state, variant, aug_idx) -> dict[str, Tensor]` chỉ tạo đặc trưng cho khối trong S; khối ngoài S bị loại hoàn toàn khỏi tính toán, không thay bằng số 0.

Nếu trạng thái động của một khối không có (ví dụ khối không kích hoạt) thì bỏ qua. Kiểm tra bằng test rằng `step_state.time` và `load` có đúng shape `[B,P]`.

### 3.3 Từ biến thể sang tập khối S (`spec.py`)

```python
def blocks_for(variant, lam, relation_present) -> frozenset:
    S = set()
    if lam[2]:                          S.add("DEM")
    if lam[5] or lam[6] or lam[7]:      S.add("TW")
    if lam[3] or lam[4]:                S.add("PRZ")
    if relation_present:                S.add("PD")     # reset_state.relation is not None
    if lam[11] or lam[12]:              S.add("RTE")
    return frozenset(S)
```

Kết quả mong đợi cho 11 bài seen (test phải khớp, ghi vào `docs/SEEN_BLOCK_SETS.md`):

| Tập S | Bài |
| --- | --- |
| ∅ | tsp, atsp |
| {PRZ} | op, pctsp |
| {PD} | pdtsp |
| {DEM, RTE} | cvrp, acvrp, cvrpb, ocvrp |
| {DEM, TW, RTE} | cvrptw, ocvrptw |

Chỉ có **5 tập khối khác nhau** trong huấn luyện. Hệ quả đã ghi trong proposal: phần lớn biến thể unseen của URS vẫn nằm trong 5 tập này (khác nhau ở MD, L, BP, bất đối xứng, do mask và khoảng cách xử lý); chỉ PDCVRP và OPDCVRP có tập mới {PD, DEM, RTE}. Vì vậy split tổ hợp P3 (E3) là bắt buộc để kiểm tra tính module. Viết `scripts/list_unseen_block_sets.py` in số biến thể unseen theo từng tập S để chủ dự án thấy rõ.

Các thuộc tính MD, Duration Limit, Backhaul-Priority **không có khối riêng**: chúng đi qua cờ depot trong CORE và qua mask của môi trường.

### 3.4 Nhóm biến thể cho báo cáo (`eval/groups.py`)

Theo paper URS: `unseen44` là Bảng 11 của paper (44 biến thể đối xứng nghiên cứu nhiều, gồm SPCTSP); `unseen55` là Bảng 12 (55 biến thể ít khám phá). Từ báo cáo: 99 bài unseen gồm 46 đối xứng và 53 bất đối xứng, nên `unseen55 = 53 bất đối xứng + {pdcvrp, opdcvrp}`, và `unseen44 = 46 đối xứng − {pdcvrp, opdcvrp}`. Test: kích thước nhóm 44 và 55 và hợp của chúng bằng `zero_shot_list`.

**Nghiệm thu giai đoạn 2:** `tests/test_spec.py` và `tests/test_features.py` qua; `test_features` kiểm tra đặc trưng của một batch nhỏ cho mỗi bài seen khớp (trên cùng dữ liệu) với tensor đặc trưng mà `Model_LIB.py` của URS dựng ra (chỉ so các cột tương ứng, sai số < 1e-6).

## 4. Giai đoạn 3: mạng nơ-ron (`matroute/nn/`, `encoder.py`)

### 4.1 `BlockLinear`

Trọng số dày `W ∈ R^{d×d}` nhân với mặt nạ cố định (buffer). `Pa(CORE) = {CORE}`, `Pa(g) = {CORE, g}` cho khối ràng buộc g. Huấn luyện dùng ma trận dày × mặt nạ rồi lát cắt theo `active_idx(S)`; suy luận không grad có thể lát cắt trước.

```python
class BlockLinear(nn.Module):
    def __init__(self, layout, bias=True):
        W = torch.empty(layout.dim(), layout.dim()); nn.init.xavier_uniform_(W)
        self.weight = nn.Parameter(W)
        mask = torch.zeros_like(W)
        for b in layout.names:
            parents = {"CORE"} if b == "CORE" else {"CORE", b}
            for p in parents:
                mask[layout.idx(b)[:, None], layout.idx(p)[None, :]] = 1
        self.register_buffer("mask", mask)
    def forward(self, h, S):
        idx = self.layout.active_idx(S)
        w = (self.weight * self.mask)[idx][:, idx]
        return h @ w.T + (self.bias[idx] if self.bias is not None else 0)
```

Quy ước tensor: luôn giữ biểu diễn rút gọn theo `active_idx(S)`. Khởi tạo hàng của khối ràng buộc nhân hệ số nhỏ (khoảng 0.1) để lúc đầu mô hình gần với TSP-solver.

### 4.2 `BlockInstanceNorm`

Instance norm tính **riêng cho từng khối** (thống kê trên các node của cùng instance, trên chiều của khối). Không dùng LayerNorm trên toàn bộ chiều.

### 4.3 `BlockAttention`

- Mỗi khối b có `n_heads(b)` head; mỗi head chỉ dùng chiều của chính khối đó cho Q, K, V (`BlockLinear` bảo đảm điều này; khối ràng buộc đọc cả CORE nên Q/K/V của khối g là tổ hợp của h_CORE và h_g).
- Điểm: `softmax(q·k/√8 + bias_pair)`.
- `bias_pair` theo khối:
  - CORE: `−α_h · D_ij`, α_h học theo từng head. Một nửa head CORE dùng `D`, nửa còn lại dùng `D^T`. Với bài đối xứng hai nửa giống nhau. Chuẩn hóa D theo giá trị lớn nhất của instance (theo cách URS làm, `[XÁC NHẬN]` ở `Model_LIB.py`).
  - PD (chỉ khi PD ∈ S): bias từ `reset_state.relation` dạng `−β_h · R_ij`. Quy ước giá trị của R (bằng 0 khi có quan hệ, 1 khi không, theo paper) phải xác nhận ở `Model_LIB.py` `[XÁC NHẬN]`.
  - DEM, TW, PRZ, RTE: dùng cùng bias khoảng cách `−α_h · D_ij` với tham số riêng của khối (ghi vào `DECISIONS.md`).
- Đầu ra gộp theo khối rồi qua `BlockLinear`.
- N ≤ 1000: attention dày. N lớn: cờ `sparse_knn` (mục 8).

### 4.4 `BlockFFN` và lớp encoder

```
h = h + BlockAttention(BlockNorm(h))      # chọn pre-norm hoặc post-norm, giữ nhất quán
h = h + BlockFFN(BlockNorm(h))            # hidden = 4 × kích thước khối, cùng mặt nạ nhân quả
```

L = 12 lớp cho cấu hình chính; smoke dùng L = 2–3. Embedding đầu vào: nối các phép chiếu đặc trưng theo khối vào đoạn chiều tương ứng.

**Nghiệm thu giai đoạn 3 (kiểm thử đơn vị, bắt buộc qua hết, chạy local):**

1. `test_prefix_equivalence`: encoder với S = ∅ cho đầu ra CORE giống hệt (sai số < 1e-6) khi các khối ràng buộc nhận đầu vào ngẫu nhiên khác nhau.
2. `test_core_ignores_constraint_blocks`: với S bất kỳ, đạo hàm của h_CORE theo đầu vào của khối ràng buộc bằng 0.
3. `test_mask_vs_slice`: lát cắt khi suy luận và mặt nạ dày cho cùng đầu ra (sai số < 1e-5).
4. `test_asym_heads`: với ma trận khoảng cách đối xứng, hai nửa head D và D^T cho cùng điểm.
5. `test_block_norm_independence`: thay đầu vào khối TW không đổi thống kê chuẩn hóa của CORE.

## 5. Giai đoạn 4: decoder funnel (`decoder.py`, `policy.py`)

### 5.1 Truy vấn, khóa và node xuất phát

`q_t = BlockLinear([h_first, h_last, state_t])` theo khối, `state_t` theo bảng ở mục 3.2 (lấy từ `step_state`). Khóa `k_i` từ embedding node của encoder, theo khối. Khối b chỉ tham gia nếu b ∈ {CORE} ∪ S.

Node xuất phát của POMO phải **giống hệt URS** (`Model.py` L101-135 `[XÁC NHẬN]`): TSP/ATSP trajectory i bắt đầu ở node i; VRP bước 0 ở depot rồi bước 1 trajectory i chọn khách i; MD và PD theo quy tắc `pomo_size` ở mục 1. Test `test_start_nodes` so hai bước đầu của MatRoute với URS trên cùng instance.

### 5.2 Hàm funnel

```python
def funnel_logits(q, k, S, feasible, topk, zeta=50.0):
    # feasible: [B,P,N+depot] bool = (step_state.ninf_mask == 0)
    scores = {b: (q[b] @ k[b].transpose(-1,-2)) / math.sqrt(q[b].shape[-1]) for b in blocks(S)}
    cand = torch.zeros_like(feasible)
    n_feas = int(feasible.sum(-1).min().item())
    for b, s in scores.items():
        s = s.masked_fill(~feasible, float("-inf"))
        kk = min(topk[b], n_feas)
        cand.scatter_(-1, s.topk(kk, dim=-1).indices, True)
    cand &= feasible
    u = zeta * torch.tanh(sum(scores.values()))
    return u.masked_fill(~cand, float("-inf")), scores, cand
```

Quy tắc:

- Lọc node khả thi **trước** khi top-k. Nếu số node khả thi nhỏ hơn k, lấy tất cả.
- `topk[b]` mặc định: CORE 50, các khối khác 20. Khi huấn luyện ở N=100, mức FUNNEL dùng k lấy từ {16, 32, 64} cho mọi khối.
- Khi `topk[b] ≥ N` cho mọi b, `cand == feasible` và logit bằng softmax đầy đủ (test bắt buộc).
- `policy` luôn trả `scores` từng khối cho chưng cất.

### 5.3 API policy và ba mức

```python
class MatRoutePolicy(nn.Module):
    def encode(self, reset_state, S, aug_idx): ...
    def decode_step(self, step_state, S, level, topk): ...    # level ∈ {"FULL","FUNNEL","CORE_ONLY"}
    def rollout(self, env, mask_fn, S, level, topk, greedy=False):
        # trả: logp [B,P] (cộng log-prob các bước), reward [B,P] (lấy ở bước cuối),
        #      và các số hạng chưng cất cộng dồn theo bước
        ...
```

- `FULL`: softmax đầy đủ trên node khả thi, điểm = tổng điểm các khối trong CORE ∪ S.
- `FUNNEL`: logit của `funnel_logits`.
- `CORE_ONLY`: softmax chỉ dùng điểm CORE trên node khả thi. Mask của env vẫn bảo đảm khả thi nên reward hợp lệ.

Vì reward chỉ có ở bước cuối, `rollout` cộng dồn `logp` qua các bước và tích lũy số hạng KL chưng cất cùng đồ thị tính toán (ở N=100 chi phí bộ nhớ nhỏ).

**Nghiệm thu giai đoạn 4 (local):**

1. `test_funnel_equals_full`: k ≥ N, logit funnel bằng FULL (sai số < 1e-6).
2. `test_feasibility`: với mọi mức, 200 rollout ngẫu nhiên (N=20, 50) trên mỗi bài seen tạo lời giải khả thi theo env (tỷ lệ bất khả thi bằng 0, `reward` hữu hạn).
3. `test_core_only_for_tsp`: với TSP, FULL và CORE_ONLY cho cùng logit.
4. `test_start_nodes`: như mục 5.1.

## 6. Giai đoạn 5: huấn luyện (`matroute/train/`, `train_matroute.py`)

### 6.1 Vòng huấn luyện

Mỗi batch:

1. Chọn biến thể P bằng `random.choice(train_problem_list)` (P1) hoặc danh sách 22 bài (P2), như `Trainer.py`.
2. S = `blocks_for(P, ...)`.
3. Chọn mức: `FULL` 0.5, `FUNNEL` 0.3, `CORE_ONLY` 0.2; 20% epoch đầu chỉ `FULL`.
4. `env.load_problems(...)` rồi `reset()`, chạy `policy.rollout` với `pomo_size = problem_size` (cùng quy tắc `Trainer.py`), `capacity` theo từng bài `[XÁC NHẬN]`.
5. Tính loss, cập nhật.

### 6.2 Loss

```python
def train_step(batch, policy, sampler, stats, lam):
    P, S = sampler.task()
    level, topk = sampler.level()
    out = policy.rollout(env, mask_fn, S, level, topk)
    adv = (out.reward - out.reward.mean(1, keepdim=True)) / (stats[(P, level)].std + 1e-8)
    loss_rl = -(adv.detach() * out.logp).mean()
    loss_dist = out.dist_loss                      # cộng dồn theo bước
    return loss_rl + lam * loss_dist
```

- `stats`: độ lệch chuẩn chạy (EMA 0.99) của reward theo cặp (biến thể, mức), khởi tạo 1. Reward OP dương, các bài khác âm; chuẩn hóa chỉ chia cho độ lệch chuẩn, không đổi dấu.

### 6.3 Chưng cất

Tại mỗi bước của rollout `FULL` và `FUNNEL`, tính logit đầy đủ `u_full` (không lọc ứng viên) và điểm từng khối `s_b`:

```
π_full = softmax(u_full[feasible]).detach()
π_b    = softmax(zeta * tanh(s_b)[feasible])
L_dist = mean_steps( Σ_{b ∈ CORE ∪ S} KL(π_full || π_b) )
```

`λ_dist` giảm tuyến tính từ 0.1 xuống 0.02 trong 80% epoch cuối.

### 6.4 Siêu tham số khởi đầu (server)

| Mục | Giá trị |
| --- | --- |
| N huấn luyện | 100 |
| Epoch × batch × kích thước | 500 × 2000 × 128 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-6, giảm lr 10 lần ở epoch 451 |
| Số lớp, d | 12; d=192 (chính) và d=128 (so khớp tham số) |
| ζ | 50 |
| Chính xác số | bf16 cho encoder nếu GPU hỗ trợ; logit và softmax fp32 |

### 6.5 Checkpoint, log và khả năng chạy lại

Checkpoint mỗi 10 epoch (server) hoặc mỗi 100 bước (smoke) gồm: cấu hình, seed, `stats`, trạng thái optimizer và scheduler, commit hash của fork và của URS. **Phải có `--resume`** (server có thể bị ngắt). Log CSV theo biến thể: loss RL, loss chưng cất, độ dài lời giải trung bình, mức được chọn.

### 6.6 Nghiệm thu giai đoạn 5 (chỉ local, chức năng)

- `smoke_local.yaml`: loss RL và loss chưng cất hữu hạn, không NaN trong toàn bộ ≤ 300 bước, đủ 4 bài được chọn.
- **Overfit một batch cố định** (TSP N=20, 16 instance, 300 bước): độ dài lời giải trung bình giảm ít nhất 15% so với lúc khởi tạo. Ngưỡng 15% là đề xuất của mình cho kiểm tra chức năng; nếu không đạt, báo cáo đường cong và nguyên nhân nghi ngờ.
- `micro_local.yaml`: vẽ đường cong loss và độ dài lời giải theo bước cho từng bài (lưu `results/local/micro_curves.png`).
- `--resume` chạy lại từ checkpoint cho cùng kết quả (trong sai số) ở bước tiếp theo.

Các cổng chất lượng G1–G3 trong proposal **chỉ được đánh giá sau khi chủ dự án chạy pilot và đầy đủ trên server**, không phải ở bước này.

## 7. Giai đoạn 6: đánh giá (`matroute/eval/tester.py`, `test_matroute.py`)

`Tester` mới gọi `MatRoutePolicy` ở chế độ greedy với augmentation ×8 (×128 cho bất đối xứng), chọn nghiệm tốt nhất qua trajectory và augmentation, đúng giao thức `Tester.py` của URS `[XÁC NHẬN]`. Dùng `DataFinder`/`DataReader`/`LibReader` của URS để lấy dữ liệu và oracle, và cùng cách tính gap (kể cả công thức riêng cho OP). Xuất CSV: `variant, group(seen|unseen44|unseen55), gap, no_aug_gap, time, n, model, seed`. Đường dẫn dữ liệu và checkpoint đọc từ cấu hình.

Bảng ánh xạ thí nghiệm của proposal sang lệnh (để server chạy):

| Mã | Lệnh/kịch bản | Đầu ra |
| --- | --- | --- |
| E1 | `test_matroute.py --group seen` | `results/e1_seen.csv` |
| E2 | `--group unseen44` và `--group unseen55` | `results/e2_unseen.csv` |
| E3 | train theo split P3 rồi `--group holdout_pairs` | `results/e3_compositional.csv` |
| E4 | `--synthetic_large --sizes 1000 2000 3000 4000 5000 7000` | `results/e4_large.csv` |
| E5 | bản `Tester_Bench`-tương đương cho Set-X, Set-AGS | `results/e5_bench.csv` |
| E6 | `--funnel_sweep` | `results/e6_funnel.csv` |
| E7 | `--prefix_eval` | `results/e7_prefix.csv` |
| E8 | `configs/ablation/*.yaml` | `results/e8_ablation.csv` |
| E9 | `scripts/analysis_*.py` | `results/e9_*.csv` |

**Nghiệm thu giai đoạn 6 (local):**

- `--max_instances 64` chạy hết cho 4 bài (TSP, CVRP, CVRPTW, PDTSP) với mô hình smoke chưa huấn luyện kỹ: pipeline ra file CSV đúng cột, gap hữu hạn.
- **Kiểm tra tính nhất quán của harness:** chạy `Tester` của MatRoute và `Tester` gốc của URS trên cùng 64 instance và cùng checkpoint URS đã tải (qua một policy bọc URS), kết quả gap giống nhau trong sai số. Phần này chứng minh harness đo đúng, không đánh giá mô hình MatRoute.
- Quy tắc báo cáo trên server: ≥ 3 seed cho cấu hình chính, mean ± std; chỉ báo "tốt hơn" khi khác biệt lớn hơn độ lệch chuẩn giữa các seed; số baseline phải có file nguồn trong `results/` hoặc ghi chú trích dẫn paper.
- Với E6, recall@K là tỷ lệ bước mà node được chính sách FULL chọn nằm trong tập ứng viên funnel, tính trên cùng trạng thái (teacher forcing theo hành động của FULL).

## 8. Giai đoạn 7: quy mô lớn và vi kiểm tra tốc độ

### 8.1 Encoder thưa

Cờ `--sparse_knn K` (mặc định 100): mỗi node chỉ attend K láng giềng gần nhất theo khoảng cách cộng một số token toàn cục (mặc định 4). Test: với K ≥ N bản thưa cho đầu ra bằng bản dày (sai số < 1e-5).

### 8.2 Vi kiểm tra tốc độ trên local (không cần huấn luyện)

`scripts/bench_funnel.py` dùng **trọng số khởi tạo ngẫu nhiên**, dữ liệu sinh ngẫu nhiên, N ∈ {200, 500, 1000, 2000} (dừng ở cỡ mà bộ nhớ GPU local chịu được; ghi nhận OOM nếu gặp), đo:

- thời gian một bước decoder và bộ nhớ đỉnh của FULL so với FUNNEL,
- thời gian encoder dày so với thưa.

Mục đích chỉ để xác nhận đường mã chạy và có tín hiệu tốc độ sơ bộ. **Không rút ra kết luận về chất lượng.** Báo cáo encoder và decoder tách riêng để không claim sai.

### 8.3 Trên server

N đến 7000 chạy trên server theo E4/E5; có thể cần chia batch theo instance cho N ≥ 3000.

## 9. Phương án dự phòng đã chuẩn bị sẵn trong mã

- `--soft_core`: lõi cổng mềm. Ở một số lớp trên (mặc định 4 lớp cuối) cho CORE đọc khối ràng buộc qua trọng số nhân cổng `tanh(g)`, g khởi tạo 0. Có test cho thấy cổng bằng 0 tái tạo đúng bản chỉ-đọc.
- `--stopgrad_core`: chặn gradient từ khối ràng buộc về CORE (ablation).
- `--flat`: biến thể Flat (mặt nạ toàn 1, LayerNorm toàn cục, không chưng cất), baseline nội bộ.
- `--slim`: biến thể slimmable chuẩn cùng số tham số, cắt theo chiều rộng.
- `--core_ladder 32,64,80`: thang chiều lồng trong CORE (mở rộng tùy chọn).

## 10. Giai đoạn 8: gói chạy server (sản phẩm cuối của agent)

Agent không chạy các lệnh dưới đây mà chuẩn bị sẵn để chủ dự án chạy trên server:

- `docs/SERVER_RUNBOOK.md`: cách dựng môi trường (phiên bản Python/PyTorch đã dùng ở local), cách đặt dữ liệu và checkpoint, danh sách lệnh theo thứ tự (pilot, full d=128, full d=192, E1–E9, ablation) kèm thời gian ước tính (từ proposal: khoảng 95 giờ cho d=128 và khoảng 140 giờ cho d=192 trên một RTX 4090; là ước tính cần hiệu chỉnh), cách `--resume`.
- `scripts/run_pilot.sh`, `run_full_d128.sh`, `run_full_d192.sh`, `run_eval_all.sh`, `run_ablation.sh` (kèm bản `.bat` hoặc ghi chú nếu server là Windows).
- Quy tắc quyết định cổng **G1–G3** (ngưỡng ở mục 9 của proposal; là đề xuất, chủ dự án xác nhận trước khi chạy): file `docs/GATES.md` nêu rõ cần so sánh những gì, đọc từ file CSV nào, và khi không đạt thì làm gì (lõi cổng mềm, hạ trọng tâm xuống funnel...).
- Danh sách kiểm tra trước khi chạy dài: tests qua hết, smoke qua, đường dẫn dữ liệu đúng, GPU đủ bộ nhớ, ổ đĩa đủ chỗ cho checkpoint, seed.

## 11. Cấu trúc thư mục đích

```
matroute/
  layout.py  spec.py  features.py
  nn/{block_linear,block_norm,block_attn,block_ffn}.py
  encoder.py  decoder.py  policy.py
  train/{sampler,nested_reinforce}.py
  eval/{tester,groups}.py
  configs/{smoke_local,micro_local,pilot_server,full_d128,full_d192}.yaml  configs/ablation/*.yaml
  tests/
train_matroute.py
test_matroute.py
docs/{URS_INTERFACE,DECISIONS,SEEN_BLOCK_SETS,ENV_LOCAL,SERVER_RUNBOOK,GATES,PROGRESS}.md
results/ (results/local/ cho kết quả chạy máy local)
scripts/
```

## 12. Thứ tự công việc và nghiệm thu

| Bước | Việc | Nơi chạy | Nghiệm thu |
| --- | --- | --- | --- |
| 0, 1 | Khảo sát, tái lập baseline | Xong | Chủ dự án đã xác nhận |
| 2 | Layout, spec, features | Local | `test_spec`, `test_features` qua; `SEEN_BLOCK_SETS.md` khớp mục 3.3 |
| 3 | Encoder block-causal | Local | 5 test đơn vị qua |
| 4 | Decoder funnel và policy | Local | 4 test qua, tỷ lệ bất khả thi bằng 0 |
| 5 | Trainer và smoke | Local | Mục 6.6 |
| 6 | Tester và kiểm tra harness | Local | Mục 7 |
| 7 | Encoder thưa và vi kiểm tra tốc độ | Local | Mục 8 |
| 8 | Gói chạy server | Local (chuẩn bị) | Runbook, scripts, gates đầy đủ |
| 9 | Pilot, huấn luyện đầy đủ, E1–E9 | **Server (chủ dự án)** | Cổng G1–G3 |

## 13. Những lỗi dễ gặp

- Đưa cờ pickup/delivery của bài VRP vào khối PD: PD chỉ dành cho bài có `relation`. Cờ của VRP và backhaul thuộc DEM.
- Parse tên bài để biết open route: dùng λ thay vì `'o' in name`.
- Quên sinh η mới cho từng bản sao augmentation ×128 của bài bất đối xứng.
- Trộn chiều giữa các khối qua LayerNorm hoặc `nn.Linear` dày không mặt nạ: phá tính prefix, test 1 và 5 sẽ báo.
- Quên lọc node khả thi trước top-k: shortlist chọn nhầm node bị mask, logit thành `-inf` toàn bộ.
- Đưa số 0 thay cho khối không kích hoạt: làm bài TSP phụ thuộc vào khối ràng buộc. Khối ngoài S phải bị loại khỏi tính toán.
- Dùng `ninf_mask` như bool: nó là `0/-inf` cộng vào logit. Chuyển bằng `(ninf_mask == 0)`.
- Trên Windows: đường dẫn, `num_workers>0`, và `torch.compile` hay gây lỗi; local mặc định fp32 và `num_workers=0`.
- So sánh với số của paper trên dữ liệu test khác với dữ liệu URS đã phát hành.
- Chạy ablation với lịch huấn luyện khác nhau giữa các biến thể: mọi ablation dùng cùng lịch rút gọn 250 epoch.

## 14. Cách báo cáo cho chủ dự án

Sau mỗi bước ở mục 12, viết ngắn vào `docs/PROGRESS.md`: bước nào xong, tiêu chí nghiệm thu đạt hay không (kèm số liệu), quyết định mới trong `DECISIONS.md`, và việc đang bị chặn. Nếu một tiêu chí không đạt, nêu giả thuyết nguyên nhân nhưng không tự đổi thiết kế mà chưa hỏi.
