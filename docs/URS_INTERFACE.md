# Khảo sát giao diện kiến trúc và môi trường URS (URS Interface Survey)

Tài liệu này tổng hợp chi tiết kết quả khảo sát mã nguồn dự án **URS (A Unified Neural Routing Solver for Cross-Problem Zero-Shot Generalization)** - ICML 2026, phục vụ giai đoạn 0 (Nghiệm thu giao diện và môi trường). Mọi câu trả lời đều đi kèm vị trí file và số dòng mã nguồn cụ thể.

---

## 1. Giấy phép sử dụng (License & Derivative Works)

- **File**: [`LICENSE`](file:///d:/URS/LICENSE#L1-L25), [`args.py`](file:///d:/URS/args.py#L4-L17)
- **Loại License**: **GNU Affero General Public License Version 3 (AGPLv3)**, ban hành ngày 19/11/2007.
- **Bản quyền**: Copyright (C) 2026 CIAM Group, Southern University of Science and Technology, Shenzhen, China.
- **Quy định về tái phân phối mã dẫn xuất**:
  - **CÓ** cho phép sửa đổi và tái phân phối mã dẫn xuất (derivative works).
  - **Điều kiện Copyleft nghiêm ngặt của AGPLv3**: Bất kỳ sản phẩm dẫn xuất nào khi phân phối, hoặc khi được triển khai chạy trên máy chủ mạng (network server) cho người dùng tương tác từ xa, **bắt buộc phải công khai toàn bộ mã nguồn** dưới cùng giấy phép AGPLv3 (`LICENSE` Điều khoản 13).
  - Không được phép biến mã nguồn dự án thành phần mềm độc quyền (proprietary closed-source). Mọi bản phân phối đều phải giữ nguyên thông báo bản quyền gốc và tuyên bố từ chối bảo hành.

---

## 2. Môi trường mô phỏng: `env/UniVRPEnv.py`

### 2.1. Cách khởi tạo (Tham số, tên bài toán, số node, số quỹ đạo)
- **Hàm tạo `__init__(self)`**: [`env/UniVRPEnv.py#L109-L178`](file:///d:/URS/env/UniVRPEnv.py#L109-L178)
  - Không nhận tham số bắt buộc. Chỉ thiết lập trạng thái rỗng và khởi tạo các đối tượng chứa trạng thái: `Reset_State()` (dòng 174) và `Step_State()` (dòng 175).
- **Hàm nạp dữ liệu và cấu hình `load_problems(...)`**: [`env/UniVRPEnv.py#L179-L308`](file:///d:/URS/env/UniVRPEnv.py#L179-L308)
  - `batch_size`: Số lượng bài toán (instances) trong một batch.
  - `problem_name`: Tên định danh biến thể VRP (ví dụ: `'tsp'`, `'cvrp'`, `'cvrptw'`, `'op'`,...).
  - `problem_size`: Số lượng nút khách hàng ($N$, không bao gồm depot).
  - `pomo_size`: Số lượng quỹ đạo (trajectories) rollout song song trên mỗi instance. Mặc định nếu `pomo_size is None`, `pomo_size = problem_size` (dòng 197-200).
  - `capacity`: Sức chứa của phương tiện (bắt buộc khi bài toán thuộc họ VRP, dòng 221-222).
  - `device`: Thiết bị tính toán PyTorch (`cpu` hoặc `cuda:X`).
  - `aug_factor`: Hệ số tăng cường dữ liệu (mặc định 1, kiểm thử thường là 8 cho đối xứng và 128 cho bất đối xứng).
  - `lib_data` / `validation_data`: Dữ liệu có sẵn nếu nạp từ file benchmark/tập kiểm thử thay vì sinh ngẫu nhiên.
- **Xử lý đặc thù số depot (`depot_num`) và số quỹ đạo (`pomo_size`)**:
  - Đối với `tsp`, `atsp`: `depot_num = 0` (dòng 189-190).
  - Đối với `md` (Multi-depot VRP): `depot_num = 3`, `pomo_size = problem_size * depot_num` (dòng 191-194, 204-205).
  - Đối với các biến thể có 1 depot: `depot_num = 1` (dòng 194).
  - Đối với bài toán `pd` (Pickup and Delivery): nếu `pomo_size > problem_size // 2` thì tự động giới hạn `pomo_size = problem_size // 2` để chỉ xuất phát từ các điểm lấy hàng (dòng 201-203).

### 2.2. Giao diện `reset()` và `step()`
- **Hàm `reset()`**: [`env/UniVRPEnv.py#L389-L471`](file:///d:/URS/env/UniVRPEnv.py#L389-L471)
  - **Nhận vào**: Chỉ nhận `self`.
  - **Trả về**: Tuple `(reset_state, reward, done)`
    - `reset_state` (instance của `Reset_State`, dòng 37-45):
      - `reset_state.problems`: Tensor shape `[batch_size, problem_size + depot_num, 9]` gồm các cột thuộc tính: $[x, y, \text{demand}, \text{prize}, \text{fake\_prize}, \text{penalty}, \text{tw\_start}, \text{tw\_end}, \text{service\_time}]$.
      - `reset_state.dist`: Tensor ma trận khoảng cách, shape `[batch_size, problem_size + depot_num, problem_size + depot_num]`.
      - `reset_state.log_scale`: Giá trị vô hướng $\log_2(\text{problem\_size})$ (dòng 427).
      - `reset_state.relation`: Tensor ma trận ghép cặp pickup-delivery shape `[batch, problem_size + 1, problem_size + 1]` cho bài toán PD, hoặc `None`.
    - `reward`: `None` (dòng 468).
    - `done`: `False` (dòng 469).

- **Hàm `step(selected, mask_fn)`**: [`env/UniVRPEnv.py#L486-L598`](file:///d:/URS/env/UniVRPEnv.py#L486-L598)
  - **Nhận vào**:
    - `selected`: Tensor kiểu `torch.long`, shape `[batch_size, pomo_size]`, chứa chỉ số của nút được chọn tiếp theo cho từng quỹ đạo.
    - `mask_fn`: Hàm callable kiểm tra tính khả thi, được tạo từ `mask_registry.build_combined_mask(problem_name)`.
  - **Trả về**: Tuple `(step_state, reward, done)`
    - `step_state` (instance của `Step_State`, dòng 47-67):
      - `step_state.current_node`: Tensor shape `[batch, pomo]`.
      - `step_state.load`: Tensor tải trọng còn lại, shape `[batch, pomo]`.
      - `step_state.time`: Tensor thời điểm hiện tại, shape `[batch, pomo]`.
      - `step_state.length`: Tensor độ dài chặng hiện tại, shape `[batch, pomo]`.
      - `step_state.tour_maxlength`: Tensor độ dài còn lại (cho OP), shape `[batch, pomo]`.
      - `step_state.collected_prize`: Tensor giải thưởng đã thu thập (cho PCTSP), shape `[batch, pomo]`.
      - `step_state.ninf_mask`: Tensor mặt nạ khả thi cho bước tiếp theo, shape `[batch, pomo, problem_size + depot_num]`.
      - `step_state.finished`: Tensor boolean shape `[batch, pomo]`.
    - `reward`:
      - Khi chưa hoàn thành (`done == False`): trả về `None` (dòng 596).
      - Khi đã hoàn thành (`done == True`): Tensor shape `[batch_size, pomo_size]`.
    - `done`: Boolean giá trị đơn (`self.finished.all()`, dòng 582).

- **Nguồn gốc và Shape của Feasibility Mask**:
  - Mask được tính toán động tại dòng 568: `self.ninf_mask = mask_fn(self)`.
  - Sau đó được chuẩn hóa nút depot qua: `self.ninf_mask = self.process_depot_specially(self.ninf_mask)` (dòng 569, L705-L737).
  - **Shape của Mask**: `[batch_size, pomo_size, problem_size + depot_num]`.
  - **Giá trị của Mask**:
    - $0.0$: Nút hợp lệ / khả thi (feasible).
    - $-\infty$ (`-float('inf')`): Nút không hợp lệ / vi phạm ràng buộc (bị cấm chọn).

### 2.3. Quy ước Reward
- **Thời điểm trả**: Reward chỉ được trả một lần duy nhất tại bước kết thúc khi toàn bộ các quỹ đạo đã xong (`if done: ... else: reward = None`, dòng 582-596).
- **Dấu của Reward**:
  - **Âm khoảng cách (Negative Cost)** đối với bài toán định tuyến chuẩn:
    - VRP thông thường: `reward = -self._get_total_distance()` (dòng 594).
    - Multi-depot VRP: `reward = -self._get_md_reward()` (dòng 589).
    - Dữ liệu Benchmark TSPLIB/CVRPLIB: `reward = -self._get_travel_distance_lib()` (dòng 592).
    - PCTSP / SPCTSP: `reward = -self._get_penalty_travel_distance()` (âm của tổng quãng đường + phạt penalty cho các điểm bỏ qua, dòng 585).
  - **Dương phần thưởng (Positive Prize)** đối với bài toán Orienteering Problem:
    - OP / AOP: `reward = self._get_prize()` (tổng điểm thưởng thu được, dòng 587).

### 2.4. Mức độ độc lập với thư mục `model/`
- Trong file `env/UniVRPEnv.py` **hoàn toàn KHÔNG import bất kỳ module nào từ thư mục `model/`**.
- Môi trường hoàn toàn tách biệt (decoupled), đóng vai trò là một Gym-like environment chuẩn cho bài toán định tuyến.
- Duy nhất dòng 72 trong docstring có đề cập chữ "Deep RL models" dưới dạng văn bản chú thích.

### 2.5. Trạng thái động được theo dõi và khả năng truy cập
- Các biến trạng thái động chính:
  1. `self.load`: Sức chứa còn lại của xe (`[batch, pomo]`, dòng 402, 524-530).
  2. `self.current_time`: Thời gian tích lũy hiện tại tính theo tốc độ và thời gian phục vụ (`[batch, pomo]`, dòng 411, 562-565).
  3. `self.collected_prize`: Tổng điểm thưởng đã thu thập (`[batch, pomo]`, dòng 407, 546).
  4. `self.tour_maxlength`: Quãng đường tối đa còn lại trước khi xe phải về trạm (`[batch, pomo]`, dòng 405, 539).
  5. `self.length`: Quãng đường đã đi trong chặng hiện tại (`[batch, pomo]`, dòng 409, 541).
  6. `self.current_node`: Chỉ số nút hiện tại (`[batch, pomo]`, dòng 391, 493).
  7. `self.selected_node_list`: Chuỗi lịch sử các nút đã đi qua (`[batch, pomo, step]`, dòng 397, 495).
  8. `self.finished`: Cờ đánh dấu quỹ đạo đã hoàn thành (`[batch, pomo]`, dòng 418, 580).
- **Khả năng truy cập từ bên ngoài**:
  - Truy cập trực tiếp qua thuộc tính của môi trường (`env.load`, `env.current_time`, `env.selected_node_list`,...).
  - Truy cập thông qua đối tượng trả về `Step_State` tại mỗi bước (`step_state.load`, `step_state.time`, `step_state.collected_prize`, `step_state.current_node`, `step_state.ninf_mask`,...).

---

## 3. Sinh dữ liệu và Augmentation: `problem/ProblemDef.py`

### 3.1. Dữ liệu trả về từ hàm sinh instance
- **Hàm `get_random_problems(...)`**: [`problem/ProblemDef.py#L25-L152`](file:///d:/URS/problem/ProblemDef.py#L25-L152)
- Trả về một **`dict`** chuẩn hóa gồm 10 keys:
  1. `'xy'`: Tọa độ $2D$ `[batch, problem_size + depot_num, 2]`. (Bài toán bất đối xứng điền số $0$).
  2. `'demand'`: Nhu cầu tải trọng `[batch, problem_size + depot_num]`.
  3. `'dist'`: Ma trận khoảng cách `[batch, problem_size + depot_num, problem_size + depot_num]`. (Tính bằng Euclid `cdist` hoặc sinh ma trận bất đối xứng).
  4. `'prize'`: Điểm thưởng từng nút `[batch, problem_size + depot_num]`.
  5. `'penalty'`: Điểm phạt bỏ nút `[batch, problem_size + depot_num]`.
  6. `'fake_prize'`: Điểm thưởng quan sát cho bài toán ngẫu nhiên SPCTSP `[batch, problem_size + depot_num]`.
  7. `'service_time'`: Thời gian phục vụ tại từng nút `[batch, problem_size + depot_num]`.
  8. `'tw_start'`: Thời điểm bắt đầu cửa sổ thời gian ($e$) `[batch, problem_size + depot_num]`.
  9. `'tw_end'`: Thời điểm kết thúc cửa sổ thời gian ($l$) `[batch, problem_size + depot_num]`.
  10. `'route_limit'`: Giới hạn tổng độ dài hành trình `[batch]`.

### 3.2. Quản lý cờ Pickup/Delivery, Open Route, Sub-route
- **Cờ Pickup / Delivery**:
  - Với PDTSP / PDCVRP: Quy ước nửa đầu $[1, N/2]$ là nút lấy hàng (Pickup), nửa sau $[N/2 + 1, N]$ là nút giao hàng (Delivery) (`ProblemDef.py#L180-L185`, `Model_LIB.py#L244-L246`, `UniVRPEnv.py#L434-L450`).
  - Với VRPB (Backhaul): Nút có `demand > 0` là Linehaul (Delivery), nút có `demand < 0` là Backhaul (Pickup) (`Model_LIB.py#L266-L267`, `UniVRPEnv.py#L550-L559`).
- **Cờ Open Route**:
  - Không nằm trong dict trả về của instance, mà được gắn trong `UniVRPEnv.py#L285-L286`: `if 'o' in self.problem_name: self.open_route = True`.
- **Sub-route**:
  - Quản lý động qua `self.current_route_num` và `self.at_the_depot` trong `UniVRPEnv.py#L394, L500-L503` (mỗi lần xe quay về trạm, số thứ tự chặng tăng lên và tải trọng được reset về 1.0).

### 3.3. Cách gọi hàm Augmentation ×8 và ×128
- **Augmentation ×8 (Bài toán đối xứng - Symmetric)**:
  - Hàm `augment_xy_data_by_8_fold(problems)`: [`problem/ProblemDef.py#L154-L173`](file:///d:/URS/problem/ProblemDef.py#L154-L173)
  - Biến đổi tọa độ $x, y$ theo 8 phép đối xứng dihedral: $(x, y)$, $(1-x, y)$, $(x, 1-y)$, $(1-x, 1-y)$, $(y, x)$, $(1-y, x)$, $(y, 1-x)$, $(1-y, 1-x)$.
  - Được gọi tại `env/UniVRPEnv.py#L243-L244` khi `aug_factor == 8` và `"a" not in self.problem_name`.
- **Augmentation ×128 (Bài toán bất đối xứng - Asymmetric)**:
  - Do bài toán bất đối xứng không có tọa độ $x, y$, batch được nhân bản 128 lần bằng `xy.repeat(aug_factor, 1, 1)` và lặp các ma trận khoảng cách (`UniVRPEnv.py#L245-L260`).
  - Khi đưa vào mô hình tại `model/Model_LIB.py#L212-L220` (`unified_node_position_construction`), một vector chiếu ngẫu nhiên `node_random_emb = torch.rand((batch_size, problem_size, 1))` được tạo mới cho từng bản sao, tạo ra 128 biểu diễn không gian khác nhau để tìm nghiệm tốt nhất.
  - Cấu hình qua tham số `--aug_factor 8 128` (`args.py#L49-L50`) và kích hoạt trong `Tester.py#L102-L103, L166`.

---

## 4. Biểu diễn đặc trưng bài toán: `problem/ProblemRep.py`

### 4.1. Định nghĩa 13 vị trí của vector $\lambda$
- **File**: [`problem/ProblemRep.py#L28-L50`](file:///d:/URS/problem/ProblemRep.py#L28-L50)
- Thứ tự 13 thuộc tính nhị phân:
  - Vị trí `0`: `identifier` - $1$ nếu dùng node ID/phép chiếu ngẫu nhiên (bất đối xứng), $0$ nếu có tọa độ.
  - Vị trí `1`: `coordinates` - $1$ nếu có dữ liệu tọa độ 2D.
  - Vị trí `2`: `demand` - $1$ nếu có ràng buộc sức chứa xe / nhu cầu hàng hóa.
  - Vị trí `3`: `prize` - $1$ nếu có phần thưởng tại nút.
  - Vị trí `4`: `penalty` - $1$ nếu có điểm phạt khi bỏ qua nút.
  - Vị trí `5`: `early_time` - $1$ nếu có cửa sổ thời gian đến sớm nhất.
  - Vị trí `6`: `late_time` - $1$ nếu có cửa sổ thời gian đến trễ nhất.
  - Vị trí `7`: `service_time` - $1$ nếu có thời gian phục vụ tại nút.
  - Vị trí `8`: `depot` - $1$ nếu có trạm xuất phát/kết thúc (depot).
  - Vị trí `9`: `pickup` - $1$ nếu có thao tác lấy hàng (PDP) hoặc nhận hàng gom về (Backhaul).
  - Vị trí `10`: `delivery` - $1$ nếu có thao tác giao hàng.
  - Vị trí `11`: `multi_route` - $1$ nếu cho phép xe chạy nhiều chặng hoặc nhiều xe.
  - Vị trí `12`: `open_route` - $1$ nếu là lộ trình mở (không cần quay về trạm kết thúc).

### 4.2. Vector $\lambda$ của 11 bài toán đã huấn luyện (Seen Problems)
| STT | Bài toán | Vector $\lambda$ (13 chiều) | Các đặc trưng kích hoạt |
|---|---|---|---|
| 1 | `tsp` | `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | coordinates |
| 2 | `atsp` | `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | identifier |
| 3 | `cvrp` | `[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]` | coordinates, demand, depot, delivery, multi_route |
| 4 | `acvrp` | `[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]` | identifier, demand, depot, delivery, multi_route |
| 5 | `cvrpb` | `[0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]` | coordinates, demand, depot, pickup, delivery, multi_route |
| 6 | `cvrptw` | `[0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]` | coordinates, demand, early, late, service, depot, delivery, multi_route |
| 7 | `ocvrp` | `[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]` | coordinates, demand, depot, delivery, multi_route, open_route |
| 8 | `ocvrptw` | `[0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]` | coordinates, demand, early, late, service, depot, delivery, multi_route, open_route |
| 9 | `op` | `[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]` | coordinates, prize, depot |
| 10 | `pctsp` | `[0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]` | coordinates, prize, penalty, depot |
| 11 | `pdtsp` | `[0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]` | coordinates, depot, pickup, delivery |

---

## 5. Danh mục bài toán: `problem/ProblemSet.py`

- **Cách lấy danh sách 11 bài seen**:
  - Truy cập `ProblemSet.get(name="train_problem_list")` hoặc `ProblemSet.train_problem_list` ([`ProblemSet.py#L140-L142`](file:///d:/URS/problem/ProblemSet.py#L140-L142)).
- **Cách lấy danh sách các bài unseen**:
  - Tổng số bài toán đánh giá là **110 bài** (`all_evaluated_list`, dòng 145-153).
  - Danh sách Zero-shot (unseen) gồm **99 bài** (`zero_shot_list`, dòng 57-63, 105, 159).
  - Phân loại theo tính đối xứng:
    - **55 bài đối xứng**: `ProblemSet.get(name="symmetric_list")` (dòng 108, 159). Trong đó có 9 bài seen và **46 bài unseen đối xứng** (hoặc 44 bài nếu phân nhóm theo các biến thể CVRP tiêu chuẩn).
    - **55 bài bất đối xứng**: `ProblemSet.get(name="asymmetric_list")` (dòng 109, 159). Trong đó có 2 bài seen (`atsp`, `acvrp`) và **53 bài unseen bất đối xứng**.
  - Truy xuất linh hoạt qua: `ProblemSet.get(name="zero_shot_list")`, hoặc bộ lọc tiền tố: `ProblemSet.get(included="a")`, `ProblemSet.get(excluded="a")` (dòng 181-214).
- **Tên biến thể dùng làm khóa**:
  - Dùng chuỗi string chữ thường ghép từ các tiền tố và hậu tố:
    `[a] [md] [o] cvrp [b | bp] [l] [tw]` (ví dụ: `cvrp`, `acvrptw`, `mdcvrpbl`, `opdcvrp`).

---

## 6. Huấn luyện củng cố: `Trainer.py`

- **Cách chọn bài mỗi batch**:
  - [`Trainer.py#L237-L238`](file:///d:/URS/Trainer.py#L237-L238): Tại mỗi batch trong số `batches_per_epoch` (2000 steps), một loại bài toán được chọn ngẫu nhiên đồng đều:
    `problem_name = random.choice(self.train_problem_list)`
  - Môi trường và mask tương ứng được khởi tạo mới hoàn toàn cho từng batch (`Trainer.py#L273-L277`).
- **Cách tạo $n$ quỹ đạo POMO (Multiple Optima)**:
  - Thiết lập `pomo_size = problem_size` (`Trainer.py#L279`).
  - Tại bước đầu tiên, mô hình gán các điểm xuất phát khác nhau cho $n$ quỹ đạo (`model/Model.py#L101-L135`):
    - TSP/ATSP: Quỹ đạo $i$ bắt đầu tại nút $i$ ($i \in [0, N-1]$).
    - VRP: Bước 0 tất cả ở depot, bước 1 quỹ đạo $i$ rẽ sang nút khách hàng $i$.
- **Cơ chế Baseline và Loss**:
  - POMO Shared Baseline ([`Trainer.py#L307-L314`](file:///d:/URS/Trainer.py#L307-L314)):
    $$\text{Advantage}_i = R_i - \frac{1}{M}\sum_{j=1}^M R_j$$
    $$\mathcal{L} = - \frac{1}{B \cdot M} \sum_{b, i} \text{Advantage}_{b, i} \sum_{t} \log \pi_\theta(a_t | s_t)$$
- **Ghi Log và Lưu Checkpoint**:
  - Ghi log ra console và file thông qua `logging.getLogger('trainer')` (`Trainer.py#L53`).
  - Vẽ đồ thị train loss và lưu tại `<result_folder>/latest.png` sau mỗi epoch (`Trainer.py#L211-L213`).
  - Checkpoint tốt nhất trên tập validation được lưu tại `<result_folder>/best_checkpoint.pt` (`Trainer.py#L190-L197`).
  - Checkpoint định kỳ lưu tại `<result_folder>/checkpoint_{epoch}.pt` (`Trainer.py#L215-L224`).

---

## 7. Đánh giá và Benchmark: `data/DataFinder.py`, `DataReader.py`, `LibReader.py`, `Tester.py`, `Tester_Bench.py`

- **Cách tìm dữ liệu test và kết quả Oracle**:
  - `DataFinder` ([`data/DataFinder.py#L162-L260`](file:///d:/URS/data/DataFinder.py#L162-L260)): Quét thư mục con theo tên bài toán (`dataset/<problem_name>/`), sử dụng regex nhận diện tên file dữ liệu và file nghiệm oracle dựa theo scale.
  - Bảng tra cứu `OracleDict` ([`data/DataFinder.py#L47-L90`](file:///d:/URS/data/DataFinder.py#L47-L90)):
    - `cvrp`: `hgs`
    - `tsp`, `atsp`, `pdtsp`, `cvrpl`, `ocvrp`: `lkh`
    - `cvrptw`, `acvrp`, `mdvrpmix`: `pyvrp`
    - Các biến thể phối hợp khác: `ortools`
    - `op`: `compass`
    - `pctsp`, `spctsp`: `ils`
- **Định dạng file**:
  - Dữ liệu tổng hợp: `.pkl`, `.pt`, `.txt` (`DataFinder.py#L211`, `DataReader.py#L26-L60`).
  - Benchmark chuẩn: `.tsp` cho TSPLIB, `.vrp` và `.sol` cho CVRPLIB (`LibReader.py#L20-L114`).
- **Cách tính Gap**:
  - Trong [`Tester.py#L283-L288`](file:///d:/URS/Tester.py#L283-L288):
    - Bài toán cực tiểu hóa: $\text{Gap} = \frac{\text{Score}_{\text{model}} - \text{Score}_{\text{oracle}}}{\text{Score}_{\text{oracle}}} \times 100\%$
    - Bài toán cực đại hóa (`op`): $\text{Gap} = \frac{\text{Score}_{\text{oracle}} - \text{Score}_{\text{model}}}{\text{Score}_{\text{oracle}}} \times 100\%$
  - Báo cáo cả 2 chỉ số: `no_aug_gap` và `aug_gap`.
- **Cách báo cáo thời gian**:
  - Dùng `TimeEstimator` (`utils/utils.py`) và `time.time()`.
  - Hiển thị: Thời gian đã chạy (`Elapsed`), thời gian còn lại dự kiến (`Remain`), tổng thời gian bài toán (`Total time`), thời gian trung bình mỗi instance (`Avg time per episode`), và xuất file tổng kết Excel `.xlsx` (`Tester.py#L274-L304, L434-L485`).

---

## 8. Tình trạng `dataset/` và `pretrained/`

- **Hiện trạng trên ổ đĩa**:
  - Thư mục `dataset/` hiện **chỉ có duy nhất** file [`dataset/README.md`](file:///d:/URS/dataset/README.md).
  - Thư mục `pretrained/` hiện **chỉ có duy nhất** file [`pretrained/README.md`](file:///d:/URS/pretrained/README.md).
- **Tính đầy đủ**:
  - **KHÔNG có sẵn dữ liệu test và checkpoint**.
  - Để chạy đầy đủ 110 bài toán với kết quả so sánh oracle, người dùng cần tải bộ dataset và checkpoint từ Google Drive theo đường dẫn trong hai file README.
  - Tuy nhiên, để kiểm tra môi trường hoặc huấn luyện lại từ đầu, hệ thống có thể tự sinh dữ liệu ngẫu nhiên on-the-fly hoàn toàn độc lập thông qua `problem/ProblemDef.py`.
