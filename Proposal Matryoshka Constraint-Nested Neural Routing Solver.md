# Proposal: Matryoshka Constraint-Nested Neural Routing Solver

Nov 5, 2026 · @Long

## 1. Tóm tắt và Motivation

**MatRoute** là một bộ giải routing đa bài toán dùng Matryoshka Representation Learning (MRL) làm nguyên lý thiết kế, không phải một loss phụ. Một vector nhúng duy nhất được chia thành các khối chiều lồng nhau theo ràng buộc: khối lõi hình học (TSP) rồi thêm khối demand, time window, prize, pickup-delivery và route-rule. Mỗi tập con khối là một bộ giải hợp lệ, và cùng cấu trúc đó được dùng ở decoder để shortlist node theo kiểu coarse-to-fine.

**Vì sao hướng này đáng làm.** Các bộ giải đa bài toán hiện nay có ba điểm yếu mà benchmark URS (110 biến thể, 11 biến thể seen, 99 unseen) cho thấy rõ:

1. **Interference giữa task.** URS-MTL kém hẳn URS-STL trên nhiều bài, ví dụ ATSP 2.26% so với 0.62% và CVRP 1.81% so với 1.63% (số trong paper URS). Mọi thuộc tính dùng chung một không gian nhúng phẳng nên không có cơ chế bảo vệ phần hình học dùng chung.
2. **Zero-shot tổ hợp còn yếu.** Với các biến thể Multi-Depot, gap của MVMoE/MTPOMO/ReLD-MTL ở mức 18–66% và URS vẫn còn 15–48%. Chưa có cấu trúc biểu diễn nào buộc từng ràng buộc thành một module có thể ghép.
3. **Quy mô lớn.** Nhiều baseline OOM ở N=5000 và gap tăng mạnh khi N tăng. Bước chọn node của decoder là một bài toán retrieval trên N node, đúng loại việc mà MRL (shortlist rẻ rồi rerank đầy đủ) được thiết kế để làm.

**Câu hỏi nghiên cứu.**

- RQ1: Cấu trúc nested theo ràng buộc có cải thiện zero-shot tổ hợp và giảm khoảng cách MTL so với STL, so với biểu diễn phẳng cùng số tham số?
- RQ2: Một lõi hình học chung, chỉ đọc, có đủ mạnh để chuyển giao giữa mọi task?
- RQ3: Decoder funnel (shortlist bằng chiều thấp, rerank bằng chiều đầy đủ) có giữ chất lượng và tăng tốc rõ rệt ở N ≥ 1000?

**Định vị so với idea ban đầu.** Idea "prefix = biến thể bài toán" bị giới hạn vì ràng buộc là tổ hợp chứ không phải chuỗi. MatRoute thay prefix bằng **MRL trên lattice các tập khối** và thêm trục thứ hai (funnel decoder), nên novelty nằm ở cấu trúc, không chỉ ở việc thêm loss MRL.

## 2. Related work và khoảng trống

MatRoute khác các hướng hiện có ở chỗ nó đặt cấu trúc lồng nhau vào chính không gian nhúng và vào bước chọn node, thay vì dùng tag, adapter hay hypernetwork. Bảng dưới tóm tắt khác biệt.

| Hướng | Đại diện | Cơ chế chia sẻ | Hạn chế mà MatRoute nhắm tới |
| --- | --- | --- | --- |
| Tổ hợp ràng buộc, nhúng phẳng | MTPOMO, MVMoE, RouteFinder, CaDA, MTL-KD | Mọi thuộc tính vào một không gian chung; RouteFinder thêm global attribute embedding | Không có cấu trúc module; phạm vi bị giới hạn bởi tập thuộc tính định sẵn |
| Adapter / fine-tuning | TSP-FT, MTL-MAB, GOAL | Backbone chung và adapter theo bài | Cần fine-tune, không zero-shot |
| Biểu diễn dữ liệu thống nhất + hypernetwork | URS | Gộp đặc trưng, sinh tham số decoder theo vector tag | Baseline mạnh nhất; MatRoute không dùng các thành phần này |
| Nested / elastic width | MRL, MatFormer, slimmable networks | Tiền tố chiều hoặc lát cắt trọng số dùng được độc lập | Chưa áp dụng cho NCO; ràng buộc VRP là tổ hợp, không phải chuỗi |
| Retrieval nhiều tầng | Adaptive retrieval với MRL | Shortlist bằng chiều thấp, rerank bằng chiều đủ | Chưa dùng làm decoder cho routing quy mô lớn |
| Chọn node theo láng giềng | ELG, INViT, LEHD, BQ | Giới hạn ứng viên bằng khoảng cách | Quy tắc hình học cố định, không học được theo ràng buộc |

**Khoảng trống.** Chưa có công trình nào (theo tìm kiếm của mình, cần quét lại arXiv trước khi submit) cùng lúc: (a) cấu trúc nhúng thành các khối ràng buộc có thể bật/tắt, (b) huấn luyện từng tập con khối như một solver độc lập, và (c) dùng cùng cấu trúc đó để shortlist ở decoder. Hai công trình gần nhất cần đọc kỹ và nêu rõ khác biệt là CaDA (attention nhận biết ràng buộc) và RouteFinder (global attribute embedding), cùng Chain-of-Context và FiLMMed ở nhóm điều biến theo ràng buộc.

**Vì sao MRL, không phải slimmable hay MatFormer.** Các phương pháp đó cắt theo chiều rộng để đổi lấy tốc độ. MatRoute cắt theo ngữ nghĩa ràng buộc, và mục tiêu chính là chuyển giao và ghép module. Ablation sẽ so trực tiếp với một mạng slimmable chuẩn để chứng minh điều này.

## 3. Đóng góp chính và giả thuyết

MatRoute đề xuất ba đóng góp, mỗi đóng góp gắn với một giả thuyết có thể bác bỏ bằng thực nghiệm.

**C1. Constraint-Matryoshka trên lattice.** Vector nhúng gồm một khối lõi hình học và năm khối ràng buộc (DEM, TW, PRZ, PD, RTE). Một biến thể bài toán kích hoạt tập con khối S, và mọi S là một bộ giải hợp lệ. Cấu trúc kết nối giữa các khối là block-causal: lõi không đọc khối ràng buộc, còn khối ràng buộc đọc lõi. Đây là MRL tổng quát hóa từ chuỗi tiền tố sang poset các tập khối.

**C2. Matryoshka funnel decoder.** Bước chọn node dùng hai tầng. Tầng 0 chấm điểm toàn bộ node bằng chiều thấp của từng khối đang bật và đề cử top-k ứng viên. Tầng 1 rerank tập ứng viên bằng toàn bộ chiều. Chi phí decoder giảm từ O(N·d) xuống O(N·d₀ + K·d).

**C3. Nested RL training.** REINFORCE với POMO baseline, lấy mẫu (task, mức lồng) mỗi batch theo quy tắc sandwich, cộng chưng cất on-policy từ chính sách đầy đủ xuống các mức thấp hơn và xuống tầng shortlist.

**Giả thuyết kiểm chứng.**

| Mã | Giả thuyết | Phép đo chính | Điều kiện bác bỏ |
| --- | --- | --- | --- |
| H1 | Nested theo khối hơn nhúng phẳng cùng số tham số trên biến thể unseen | Gap trung bình 99 biến thể unseen; split tổ hợp có kiểm soát | Chênh lệch nhỏ hơn độ lệch chuẩn giữa các seed |
| H2 | Lõi chỉ-đọc giảm khoảng cách MTL so với STL | Gap seen so với STL cùng kiến trúc | Lõi chỉ-đọc kém hẳn bản có cổng mềm (xem ablation) |
| H3 | Funnel giữ chất lượng ở N lớn | Chênh gap so với softmax đầy đủ; speedup | Mất hơn 1% gap hoặc speedup dưới 1.5 lần ở N=2000 |
| H4 | Các cấu hình tiền tố cho đường cong chất lượng–chi phí mượt | Gap theo FLOPs khi tắt khối | Gap tăng đột ngột khi tắt khối không liên quan |

**Phạm vi trung thực.** MatRoute không có khối riêng cho thuộc tính chưa từng thấy lúc train (Multi-Depot, Duration Limit, Backhaul-Priority). Các thuộc tính này đi qua cờ node-type trong lõi và masking, giống cách URS xử lý, nên mình không kỳ vọng thắng URS ở mọi biến thể MD. Khẳng định cần chứng minh là lợi thế trên nhóm biến thể kết hợp các khối đã học, và lợi thế về tốc độ ở quy mô lớn.

### 3.1 Chấm novelty

**Điểm tổng thể hiện tại: khoảng 6/10.** Đây là đánh giá chủ quan của mình, dựa trên các lần tìm kiếm trước đó và nội dung paper URS; chưa phải rà soát tài liệu đầy đủ, nên cần quét lại arXiv trước khi submit.

| Thành phần | Điểm | Lý do |
| --- | --- | --- |
| C1. Constraint-Matryoshka trên lattice | 5.5/10 | Chưa thấy ai làm trong NCO, nhưng ý tưởng gần với MatFormer, slimmable networks và biểu diễn tách rời theo ràng buộc (CaDA, FiLMMed); phần mới là cấu trúc block-causal theo tập ràng buộc |
| C2. Funnel decoder | 6.5/10 | Shortlist theo láng giềng đã có (ELG, INViT); phần mới là shortlist học được, hợp đề cử theo từng khối ràng buộc, và cùng cấu trúc nesting với encoder |
| C3. Nested RL training | 4.5/10 | Sandwich sampling và chưng cất on-policy là kỹ thuật quen thuộc; đóng góp ở đây chủ yếu là kỹ thuật chứ không phải ý tưởng |
| Giao thức tổ hợp (E3) và claim zero-shot module | 6.5/10 | Benchmark URS chỉ có 6 tập khối được thấy; split tổ hợp có kiểm soát là phép đo mới và làm rõ được giá trị của cấu trúc |

**Reviewer sẽ hỏi gì.**

1. Khác gì slimmable/MatFormer áp dụng vào NCO? Trả lời bằng ablation Slim và Flat trong E8.
2. Khác gì CaDA và RouteFinder về biểu diễn ràng buộc? Cần bảng so sánh cơ chế và E3.
3. Funnel có thật sự nhanh hơn không, khi encoder vẫn O(N²)? Trả lời bằng E4 và E6, kèm encoder thưa.
4. Vì sao không thắng URS ở các biến thể MD? Đã nêu ở phần phạm vi; cần giữ claim hẹp.

**Điểm có thể thay đổi.**

| Kịch bản | Điểm kỳ vọng |
| --- | --- |
| G2 và G3 đều qua, E3 cho thấy lợi thế rõ so với Flat và URS, funnel giữ chất lượng ở N ≥ 2000 | 7–7.5/10 |
| Chỉ một trong hai (nested hoặc funnel) cho kết quả rõ | 5.5–6/10 |
| Nested chỉ hơn Flat trong phạm vi sai số seed, funnel không tăng tốc | 4–4.5/10 |

Để giữ điểm ở mức cao, ưu tiên làm E3 và E6 sớm vì đây là hai thí nghiệm quyết định novelty được chấp nhận hay bị coi là ghép kỹ thuật.

## 4. Kiến trúc chi tiết

MatRoute giữ khung encoder–decoder tự hồi quy kiểu AM/POMO, nhưng chia mọi lớp thành các khối chiều và thay bước chọn node bằng funnel hai tầng.

&#91;embedded content: kiến trúc MatRoute · CORE, 5 khối ràng buộc, funnel 2 tầng\]

CORE chỉ đọc chính nó nên TSP chạy độc lập; mỗi biến thể kích hoạt một tập khối S, và decoder lấy hợp đề cử của các khối đang bật trước khi rerank đủ chiều.

### 4.1 Giao diện dữ liệu và các khối chiều

Dữ liệu vào dùng cùng tập đặc trưng nút như benchmark URS để so sánh công bằng: tọa độ, nhiễu định danh cho bài bất đối xứng, demand (có dấu), prize, penalty, thời điểm sớm/muộn nhất và thời gian phục vụ, cờ depot/pickup/delivery, cờ open-route/sub-route. Đây là giao diện dữ liệu của benchmark, **không** phải đóng góp của bài. Mỗi nhóm đặc trưng đi qua một phép chiếu tuyến tính riêng vào khối tương ứng.

| Khối | Chiều (cấu hình d=192) | Đặc trưng vào | Biến thể seen dùng khối này |
| --- | --- | --- | --- |
| CORE | 80 | x, y, nhiễu η (bất đối xứng), cờ depot | Mọi biến thể |
| DEM | 32 | demand có dấu, cờ pickup/delivery theo dấu demand (backhaul); tải còn lại | CVRP, ACVRP, CVRPTW, CVRPB, OCVRP, OCVRPTW |
| TW | 32 | e, l, s; trạng thái thời gian hiện tại | CVRPTW, OCVRPTW |
| PRZ | 16 | prize, penalty; prize/độ dài còn lại | OP, PCTSP |
| PD | 16 | ma trận quan hệ cặp pickup–delivery; cờ pickup/delivery của bài PD | PDTSP |
| RTE | 16 | cờ open-route, sub-route | mọi bài có sub-route: CVRP, ACVRP, CVRPTW, CVRPB, OCVRP, OCVRPTW |

Mỗi khối là bội của head-dim 8 (cấu hình d=192 có 24 head: 10, 4, 4, 2, 2, 2). Cấu hình nhỏ d=128 (CORE 48, DEM 24, TW 24, PRZ 8, PD 8, RTE 16) dùng để so khớp tham số với URS.

Biến thể bất đối xứng (A) không có khối riêng: nửa số head của CORE dùng ma trận khoảng cách D, nửa còn lại dùng Dᵀ. Với bài đối xứng, hai nửa trùng nhau. Quan hệ pickup–delivery đi vào bias của head thuộc khối PD. Thuộc tính MD, Duration Limit, Backhaul-Priority chưa có khối riêng: chúng đi qua cờ node-type trong CORE và hàm masking của môi trường.

### 4.2 Lattice kích hoạt

Một biến thể có tập khối S ⊆ {DEM, TW, PRZ, PD, RTE}. Chiều kích hoạt là:

```latex
I(S) = I_{CORE} \cup \bigcup_{g \in S} I_g
```

Có 2^5 = 32 tập khối, nhưng huấn luyện theo chuẩn URS chỉ thấy 5 tập: rỗng (TSP, ATSP), {PRZ} (OP, PCTSP), {PD} (PDTSP), {DEM, RTE} (CVRP, ACVRP, CVRPB, OCVRP) và {DEM, TW, RTE} (CVRPTW, OCVRPTW). Hệ quả cần lưu ý: phần lớn biến thể unseen của URS vẫn nằm trong các tập này (khác nhau ở MD, L, BP và bất đối xứng, do mask và khoảng cách xử lý); chỉ PDCVRP và OPDCVRP có tập khối mới {PD, DEM, RTE}. Vì vậy benchmark URS một mình không đo được tính tổ hợp, và split P3 (E3) là thí nghiệm bắt buộc để kiểm chứng H1.

### 4.3 Lớp encoder block-causal

Mọi phép tuyến tính trong lớp encoder dùng trọng số có mặt nạ cấu trúc. Với vector nhúng h = \[h\_CORE, h\_DEM, h\_TW, ...\], đầu ra của khối b là:

```latex
h'_b = \sum_{b' \in \mathrm{Pa}(b) \cap S} W_{b b'} \, h_{b'}, \qquad \mathrm{Pa}(CORE) = \{CORE\},\ \mathrm{Pa}(g) = \{CORE, g\}
```

Nghĩa là CORE chỉ đọc CORE, nên TSP (S = ∅) cho cùng kết quả dù các khối khác nhận gì. Mỗi khối ràng buộc đọc CORE và chính nó. Tương tác chéo giữa các khối ràng buộc (ví dụ DEM với TW) chỉ xảy ra qua attention ở các node và qua decoder; không có trọng số trộn trực tiếp, để giữ tính module.

Các chi tiết cần đúng để tiền tố có nghĩa:

1. **Attention theo nhóm head.** Mỗi khối sở hữu một nhóm head, điểm attention của khối chỉ dùng chiều của nó (khối ràng buộc cộng thêm điểm từ CORE). Không trộn head giữa các khối.
2. **Chuẩn hóa theo khối.** Dùng instance norm riêng cho từng khối. LayerNorm trên toàn bộ chiều sẽ làm thống kê của lõi thay đổi theo tập khối kích hoạt và phá tính prefix.
3. **FFN theo khối.** Hidden 4× chiều của khối, cùng mặt nạ nhân quả.
4. **Bias cặp theo khối.** CORE dùng bias khoảng cách học được trên mỗi head; khối PD dùng bias từ ma trận quan hệ.

Số lớp encoder L = 12 để khớp URS.

### 4.4 Decoder funnel

Tại bước t, truy vấn là q\_t = \[h\_first, h\_last, state\_t\] chiếu theo khối, trong đó state\_t là trạng thái đặc thù của từng khối (tải còn lại cho DEM, thời gian hiện tại cho TW, prize còn lại cho PRZ). Khóa k\_i lấy từ embedding node.

**Tầng 0 (shortlist).** Với mỗi khối b đang bật, tính điểm chiều thấp trên các node khả thi:

```latex
s_i^{(b)} = \frac{q_t^{(b)} \cdot k_i^{(b)}}{\sqrt{d_b}}, \qquad C_t = \bigcup_{b \in \{CORE\} \cup S} \mathrm{TopK}_{k_b}\big(s^{(b)}\big)
```

Hợp các đề cử theo khối giúp node khẩn cấp theo time window vẫn được TW nêu ra dù không gần về hình học.

**Tầng 1 (rerank).** Chấm điểm đầy đủ chỉ trên C\_t và lấy softmax trên C\_t:

```latex
u_i = \zeta \tanh\Big(\sum_{b \in \{CORE\} \cup S} \frac{q_t^{(b)} \cdot k_i^{(b)}}{\sqrt{d_b}}\Big), \quad i \in C_t
```

Khi k\_b ≥ N thì C\_t là toàn bộ node khả thi và funnel trở thành softmax đầy đủ. Đây là điều kiện kiểm thử đơn vị.

### 4.5 Encoder cho N lớn

Attention đầy đủ có bộ nhớ O(N²) nên sẽ OOM ở N khoảng 5000 như các baseline (đã thấy ở bảng của URS). Ở suy luận N > 1000, encoder dùng attention thưa kNN theo khoảng cách CORE (k khoảng 100) cộng một số token toàn cục. Độ lệch train/test do thay đổi này là rủi ro đã ghi ở mục 9. Funnel chỉ giảm chi phí decoder, nên mình không claim giảm độ phức tạp tổng nếu chưa có encoder thưa.

## 5. Loss và huấn luyện

Huấn luyện là REINFORCE trên một lưới (task, mức lồng) được lấy mẫu mỗi batch, cộng chưng cất on-policy để các mức thấp và tầng shortlist học xếp hạng giống chính sách đầy đủ. Lý do không tính loss ở mọi mức trong mỗi batch là chi phí rollout nhân lên theo số mức.

### 5.1 Lấy mẫu (task, mức)

Mỗi batch chọn một biến thể P từ tập huấn luyện (như URS: một bài ngẫu nhiên mỗi batch) với tập khối S(P). Sau đó chọn mức lồng m theo quy tắc sandwich lấy cảm hứng từ slimmable networks:

- m = FULL (mọi khối của S(P), softmax đầy đủ): xác suất 0.5.
- m = FUNNEL (full nhưng qua shortlist với k ngẫu nhiên trong {16, 32, 64}): xác suất 0.3.
- m = CORE-ONLY (chỉ chạy lõi, đánh giá bằng reward của biến thể P chỉ khi lời giải vẫn khả thi nhờ masking; nếu không, chỉ dùng loss chưng cất): xác suất 0.2.

Các xác suất này là siêu tham số khởi đầu và nằm trong ablation.

### 5.2 Mục tiêu

Với trajectory τ sinh từ chính sách ở mức m, N điểm bắt đầu như POMO và baseline dùng chung:

```latex
\mathcal{L}_{RL} = -\mathbb{E}_{\tau}\Big[\big(\hat R(\tau) - \bar R\big)\, \log \pi_m(\tau)\Big], \qquad \hat R = \frac{R - \mu_P}{\sigma_P}
```

Trong đó R̄ là trung bình reward của N trajectory của cùng instance, còn μ\_P, σ\_P là thống kê reward chạy theo từng biến thể (chuẩn hóa reward đa biến thể kiểu RouteFinder) để các bài có thang reward khác nhau không lấn át nhau.

Chưng cất on-policy: trên các trạng thái mà chính sách FULL thăm, kéo các mức thấp và tầng shortlist về chính sách đầy đủ (stop-gradient):

```latex
\mathcal{L}_{dist} = \mathbb{E}_{s \sim \pi_{full}}\Big[ \sum_{b} w_b\, \mathrm{KL}\big(\mathrm{sg}[\pi_{full}(\cdot|s)] \,\|\, \pi^{(b)}(\cdot|s)\big)\Big]
```

Với π^(b) là chính sách chỉ dùng điểm chiều thấp của khối b (softmax trên node khả thi). Số hạng này vừa dạy các tiền tố, vừa huấn luyện đề cử của tầng 0, vì top-k không khả vi và không nhận gradient từ RL.

Tổng loss:

```latex
\mathcal{L} = \mathcal{L}_{RL} + \lambda_{dist}\, \mathcal{L}_{dist}
```

Các tùy chọn nghiên cứu trong ablation: thêm loss đối lập giữa biểu diễn lõi của cùng instance dưới các tập S khác nhau (khuyến khích tính bất biến của lõi), và bản lõi "cổng mềm" cho phép CORE đọc khối ràng buộc ở các lớp trên với cổng khởi tạo bằng 0.

### 5.3 Công thức huấn luyện

| Mục | Giá trị khởi đầu | Ghi chú |
| --- | --- | --- |
| Quy mô huấn luyện | N = 100 | Giống URS; kiểm tra tổng quát hóa lên N lớn |
| Epoch × batch × kích thước | 500 × 2000 × 128 | Giống URS (khoảng 95 giờ trên một RTX 4090, theo paper URS) |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-6 | Giảm lr 10 lần ở epoch 451 |
| Số lớp / d | 12 / 192 (và bản d=128) | Bản d=128 để so khớp tham số |
| Baseline | POMO shared, N điểm bắt đầu | Tăng cường instance ×8 khi suy luận (×128 cho bất đối xứng, như URS) |
| λ\_dist | 0.1 → 0.02 (giảm dần) | Chọn bằng pilot |
| Tập huấn luyện | 11 biến thể của URS (P1); 22 biến thể (P2) | P2 để so với URS-Expand |

Lịch huấn luyện: 20% epoch đầu chỉ huấn luyện mức FULL để ổn định rồi mới bật sandwich và chưng cất. Nếu bất ổn, giảm tỷ lệ CORE-ONLY và tăng λ\_dist.

## 6. Thiết lập thực nghiệm

Thiết lập bám chặt benchmark URS (cùng biến thể, cùng sinh dữ liệu, cùng oracle) để mọi con số so sánh được, và bổ sung các split mà URS không có để kiểm tra tính tổ hợp.

### 6.1 Dữ liệu và biến thể

| Nhóm | Nội dung | Số lượng |
| --- | --- | --- |
| Seen (huấn luyện) | ATSP, TSP, OP, PCTSP, PDTSP, ACVRP, CVRP, CVRPTW, CVRPB, OCVRP, OCVRPTW | 11 |
| Unseen, nghiên cứu nhiều | Tổ hợp C, O, L, TW, MD, B, BP, cộng SPCTSP | 44 |
| Unseen, ít khám phá | Kết hợp bất đối xứng (A) và PD, PC | 55 |
| Test mỗi biến thể | 1,000 instance, N = 100 | 110 biến thể |
| Quy mô lớn tổng hợp | CVRPTW, CVRPB, OCVRPTW; N = 1000–5000 (16 instance mỗi cỡ); thêm N = 7000 | 3 bài |
| Benchmark thật | CVRPLib Set-X (500 ≤ N ≤ 1000), Set-AGS (N từ 3000 đến 7000) |  |

Sinh dữ liệu theo đúng paper URS: dung lượng C = 50 (C = 20 cho PDCVRP), 20% backhaul, time window và duration theo MVMoE, 3 depot cho MD, prize/penalty theo AM. Mình sẽ dùng mã và dữ liệu gốc của URS khi có để tránh sai khác.

### 6.2 Các giao thức huấn luyện

- **P1 (chính):** huấn luyện trên 11 biến thể seen của URS. Dùng để so sánh trực tiếp với URS.
- **P2:** huấn luyện trên 22 biến thể (tập của MTPOMO/MVMoE), để so với URS-Expand.
- **P3 (split tổ hợp có kiểm soát):** giữ lại một cặp thuộc tính khỏi huấn luyện, ví dụ không bao giờ thấy đồng thời TW và Backhaul, rồi đo trên các biến thể chứa cả hai. Cần sinh thêm dữ liệu huấn luyện từ môi trường nhưng không đụng tới tập test.

### 6.3 Baseline

| Loại | Phương pháp | Cách lấy kết quả |
| --- | --- | --- |
| Đa bài toán | MTPOMO, MVMoE/4E, ReLD-MTL, GOAL-MTL | Số trong paper URS, cộng chạy lại khi cần cùng dữ liệu |
| Đa bài toán | RouteFinder (RF-TE), CaDA | Huấn luyện lại dưới cùng cấu hình (paper URS báo RF bị NaN sau 150 epoch, nên so tại 150 epoch) |
| Đa bài toán | URS | Mã công khai của tác giả; đây là đối thủ chính |
| Chuyên biệt | POMO, Sym-NCO, BQ, LEHD, ICAM, ELG, MatNet, Heter-AM | Số tham khảo cho từng bài |
| Cổ điển (oracle) | LKH-3, PyVRP, OR-Tools, ILS, Compass | Như URS: dùng để tính gap |
| Nội bộ | MatRoute-STL, MatRoute-Flat (không khối), MatRoute-Slim (slimmable chuẩn) | Chạy trong cùng khung |

### 6.4 Chỉ số

- Gap so với oracle và thời gian suy luận tổng, như URS.
- Gap trung bình theo nhóm (seen, unseen 44, unseen 55) và số biến thể tốt nhất.
- **Composition gap:** gap trên biến thể tổ hợp trừ trung bình gap của các biến thể thành phần.
- **Recall@K của shortlist:** tỷ lệ bước mà node được chính sách đầy đủ chọn nằm trong C\_t.
- Bộ nhớ đỉnh, FLOPs và thời gian trên mỗi instance, theo N.
- Ít nhất 3 seed huấn luyện cho các bảng chính; báo cáo trung bình và độ lệch chuẩn, kèm kiểm định cặp (Wilcoxon) trên instance.

### 6.5 Phần cứng

Một GPU RTX 4090 24GB cho mỗi lần huấn luyện (như URS) và CPU Xeon cho oracle. Ước tính ngân sách tính toán nằm ở mục 10.

## 7. Kế hoạch thực nghiệm đầy đủ

Chín nhóm thực nghiệm dưới đây trả lời bốn giả thuyết H1–H4 và dựng sẵn các bảng của bài báo. Mốc "mục tiêu" lấy từ số đã công bố trong paper URS để mình biết cần vượt gì; đây là tham chiếu, không phải kết quả của MatRoute.

| Mã | Thí nghiệm | Trả lời | Baseline chính | Mốc tham chiếu (URS) |
| --- | --- | --- | --- | --- |
| E1 | Hiệu năng trên 11 biến thể seen, N=100 | H2 | MVMoE, MTPOMO, ReLD-MTL, GOAL-MTL, RF-TE, CaDA, URS, STL | TSP 0.57%, CVRP 1.81%, ATSP 2.26%, CVRPTW 6.13% |
| E2 | Zero-shot 99 biến thể unseen | H1 | URS, MVMoE, MTPOMO, ReLD-MTL, RF, CaDA (RF/CaDA trên tập thuộc tính chung) | MDCVRP 15.05%, MDOCVRPBP 24.44%, CVRPBP 12.95% |
| E3 | Split tổ hợp có kiểm soát (P3) | H1 | MatRoute-Flat, URS, RF-TE | Không có số URS; tự đo |
| E4 | Quy mô lớn tổng hợp, N = 1000–7000 | H3 | URS, ReLD-MTL, MTPOMO, MVMoE, PyVRP, OR-Tools | CVRPTW N=5000: gap 8.45%, 2.3 giờ |
| E5 | Benchmark thật Set-X và Set-AGS | H3 | URS, RF-TE, MVMoE, LEHD, BQ, ELG | Set-X trung bình 8.678%; Set-AGS trung bình 13.29% |
| E6 | Phân tích funnel | H3 | Shortlist ngẫu nhiên, chỉ-lõi, kNN cố định | Không có |
| E7 | Cấu hình tiền tố và anytime | H4 | MatRoute-Slim | Không có |
| E8 | Ablation | H1–H4 | Các biến thể nội bộ | Không có |
| E9 | Phân tích biểu diễn | RQ2 | Nội bộ | Không có |

### E1. Seen

Tái hiện đúng giao thức bảng 2 và 3 của URS: gap và thời gian, tăng cường ×8 (×128 cho bất đối xứng). Báo cáo thêm MatRoute-STL (cùng kiến trúc, huấn luyện từng bài) để đo trực tiếp khoảng cách MTL so với STL. Mục tiêu: khoảng cách MTL–STL của MatRoute nhỏ hơn của URS (URS: ATSP 2.26% so với 0.62%).

### E2. Zero-shot trên 99 biến thể

Chia thành 44 biến thể nghiên cứu nhiều (có baseline) và 55 biến thể bất đối xứng/PD (chỉ URS xử lý được). Chia nhỏ kết quả theo tập khối kích hoạt để thấy: biến thể nào gồm các khối đã học thì MatRoute nên tốt hơn, biến thể nào dựa vào MD/L/BP không có khối riêng thì chỉ cần ngang URS. Mình đặt tiêu chí chấp nhận: gap trung bình thấp hơn URS ở nhóm "chỉ gồm khối đã học", và không tệ hơn đáng kể ở nhóm còn lại.

### E3. Split tổ hợp

Huấn luyện các mô hình chỉ thấy từng khối riêng lẻ hoặc theo cặp, rồi kiểm tra trên cặp bị giữ lại (TW×Backhaul, PD×Demand, PRZ×Demand, TW×Open). So MatRoute, MatRoute-Flat, và URS huấn luyện cùng tập. Đây là thí nghiệm then chốt cho H1 vì chuẩn URS chỉ có 6 tập khối được thấy.

### E4. Quy mô lớn

Theo giao thức URS: CVRPTW, CVRPB, OCVRPTW, N từ 1000 đến 5000, 16 instance mỗi cỡ, dung lượng theo TAM, tăng cường ×8; thêm N = 7000. Đo gap, thời gian, bộ nhớ đỉnh. Ở N ≥ 3000 dùng encoder kNN thưa. Báo cáo cả bản có và không có funnel để tách phần đóng góp của funnel.

### E5. Benchmark thật

CVRPLib Set-X với 500 ≤ N ≤ 1000 (tính gap so với tối ưu đã biết) và Set-AGS (Leuven1/2, Antwerp1/2). Đây là nơi baseline OOM ở N ≥ 6000, nên báo cáo rõ ô OOM.

### E6. Phân tích funnel

- Vẽ đường cong gap theo k (tổng ứng viên) cho N = 1000, 2000, 5000.
- Recall@K theo khối: dùng riêng CORE, riêng từng khối, và hợp các khối.
- So sánh bốn cách shortlist: ngẫu nhiên, kNN khoảng cách cố định, chỉ-CORE, hợp theo khối (đề xuất).
- Đo speedup và bộ nhớ của decoder theo N.
- Phân tích lỗi: bước nào node tối ưu bị loại, đặc biệt ở biến thể TW chặt.

### E7. Tiền tố và anytime

Chạy cùng một mô hình với S nhỏ dần (tắt TW, tắt DEM, chỉ CORE) trên các bài vẫn khả thi nhờ masking, và với ngân sách chiều lõi nhỏ hơn nếu bật thang chiều lõi (mở rộng tùy chọn, xem mục 9). Vẽ gap theo FLOPs và so với MatRoute-Slim.

### E8. Ablation

| Nhóm | Biến thể | Kiểm chứng |
| --- | --- | --- |
| Cấu trúc | Nhúng phẳng (Flat) / khối nhưng không nested loss / khối + nested | Giá trị của nesting (H1) |
| Kết nối | Lõi chỉ-đọc / lõi cổng mềm / mọi khối trộn đầy đủ | Giả định chỉ-đọc (H2) |
| Chuẩn hóa | Instance norm theo khối / LayerNorm toàn cục | Tính prefix |
| Loss | Bỏ chưng cất / bỏ sandwich / mức cố định / chỉ FULL | C3 |
| Funnel | Bỏ funnel / k cố định / hợp khối / chỉ CORE | C2 (H3) |
| Kích thước khối | CORE 40%, 50%, 60% của d | Phân bổ chiều |
| Slimmable | Slimmable chuẩn cùng số tham số | Nesting ngữ nghĩa so với theo chiều rộng |
| Tập huấn luyện | P1 (11) so với P2 (22) | Độ nhạy tập train |
| Cỡ mô hình | d=128 so với d=192 | Công bằng tham số |

### E9. Phân tích biểu diễn

- **Chuyển giao lõi:** CKA giữa biểu diễn CORE của cùng instance dưới các biến thể khác nhau; với lõi chỉ-đọc giá trị này bằng 1 theo cấu trúc, nên chỉ có ý nghĩa với bản cổng mềm.
- **Probe cạnh tối ưu:** huấn luyện bộ phân loại tuyến tính trên CORE đã đóng băng để dự đoán cạnh nằm trong lời giải tốt; so sánh độ chính xác giữa các task.
- **Tắt khối ở suy luận:** bật khối DEM/TW nhưng đặt đầu vào bằng 0 để đo khối đó thực sự được dùng.
- **Thêm khối mới ít mẫu:** đóng băng CORE và các khối cũ, huấn luyện chỉ một khối mới (ví dụ khối MD) trên ít epoch, so với adapter và so với huấn luyện lại. Thí nghiệm này chứng minh tính module.

### Thứ tự thực hiện

E1 trước (kiểm tra sức mạnh cơ bản), rồi E8 các ablation cấu trúc nhỏ chạy song song, rồi E2/E3, sau đó E4–E6, cuối cùng E7 và E9.

## 8. Implementation và codebase

Khuyến nghị: fork repo URS, giữ nguyên các thư mục `env/`, `problem/`, `data/`, `utils/` và thêm một gói mới `matroute/` cùng hai entry point riêng. Cách này tái dùng đúng dữ liệu, môi trường và đánh giá của URS, nên số liệu so sánh được trực tiếp với paper, trong khi mô hình, vòng huấn luyện và cấu hình của MatRoute nằm tách biệt để ablation sạch.

### 8.0 Kết quả khảo sát repo URS

Thông tin dưới đây lấy từ báo cáo của agent đã đọc repo (README, `args.py`, `ProblemDef.py`, `ProblemRep.py`, `ProblemSet.py`, `UniVRPEnv.py`, `mask_registry.py`, `Model.py`, `Model_LIB.py`, `Trainer.py`, `test.py`, `DataFinder.py`, `DataReader.py`, `LibReader.py`). Mình chưa tự đọc mã, nên chữ ký hàm và định dạng tensor phải được agent xác nhận lại ở bước đầu tiên (xem file hướng dẫn).

| Thành phần URS | Vai trò | Quyết định cho MatRoute |
| --- | --- | --- |
| `problem/ProblemSet.py` | Danh mục 110 biến thể và phân loại seen/unseen | Dùng lại nguyên vẹn |
| `problem/ProblemRep.py` | Vector 13 chiều theo biến thể (λ) | Dùng để suy ra tập khối S của MatRoute; không dùng làm đầu vào hypernetwork |
| `problem/ProblemDef.py` | Sinh instance ngẫu nhiên và tăng cường (×8, ×128) | Dùng lại; thêm adapter gói đặc trưng theo khối |
| `env/UniVRPEnv.py` | Môi trường MDP cho 110 biến thể | Dùng lại; cần kiểm tra giao diện reset/step |
| `env/mask/*` và `mask_registry.py` | Mặt nạ khả thi ghép tự động theo tên bài | Dùng lại nguyên vẹn; đây là phần giúp zero-shot cho MD, L, BP |
| `data/DataFinder.py`, `DataReader.py`, `LibReader.py` | Dữ liệu test, kết quả oracle, đọc TSPLIB/CVRPLIB | Dùng lại |
| `Tester.py`, `Tester_Bench.py`, `test.py` | Đánh giá tổng hợp và benchmark | Dùng lại logic; viết lớp Tester mới gọi policy của MatRoute |
| `model/Model.py`, `model/Model_LIB.py` (AAFM, MBM, hypernetwork) | Mô hình URS | **Không dùng** cho MatRoute; chỉ giữ để chạy baseline URS |
| `Trainer.py`, `train.py`, `args.py` | Vòng POMO/REINFORCE, tham số CLI | Tham khảo cách chọn bài mỗi batch và lưu checkpoint; viết trainer riêng |
| `pretrained/` | Checkpoint URS | Dùng để chạy baseline URS mà không huấn luyện lại |

**Đã xác nhận sau khảo sát giao diện:** giấy phép là AGPLv3 (mã dẫn xuất khi phân phối hoặc chạy như dịch vụ mạng phải công khai dưới cùng giấy phép); `UniVRPEnv` hoàn toàn không import `model/` nên dùng trực tiếp, không cần lớp bọc; giai đoạn tái lập baseline URS với dữ liệu và checkpoint tải từ Drive đã được xác nhận thành công. Hai điều chỉnh thiết kế rút ra: (1) cờ pickup/delivery của bài VRP chỉ mã hóa dấu của demand nên thuộc khối DEM, khối PD chỉ dành cho bài có ma trận quan hệ cặp; (2) huấn luyện URS chỉ có 5 tập khối khác nhau nên split tổ hợp P3 là bắt buộc (xem 4.2).

**Cách chạy:** mã được viết và kiểm tra bằng các lần chạy nhỏ trên máy local; huấn luyện và đánh giá đầy đủ chạy trên server (xem 10.1). File hướng dẫn cho agent là `MATROUTE_IMPLEMENTATION_GUIDE.md` (v2).

### 8.1 Repo liên quan

Các đường dẫn dưới đây lấy từ bảng license trong paper URS (và RL4CO từ kiến thức của mình, chưa mở kiểm tra). Cần kiểm tra giấy phép và nội dung từng repo trước khi fork.

| Mục đích | Repo | Dùng để |
| --- | --- | --- |
| Benchmark và đối thủ chính | [CIAM-Group/URS](https://github.com/CIAM-Group/URS) | Sinh dữ liệu 110 biến thể, masking, oracle, script đánh giá, baseline URS |
| Môi trường thống nhất, huấn luyện hỗn hợp | [ai4co/routefinder](https://github.com/ai4co/routefinder) | Mixed batch training, chuẩn hóa reward đa biến thể, baseline RF-TE |
| Thư viện RL cho CO | [ai4co/rl4co](https://github.com/ai4co/rl4co) | Khung huấn luyện, môi trường, POMO (xác minh phiên bản) |
| Baseline | [RoyalSkye/Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) | MVMoE, MTPOMO, dữ liệu MVMoE |
| Baseline | [FeiLiu36/MTNCO](https://github.com/FeiLiu36/MTNCO) | MTPOMO |
| Baseline | [CIAM-Group/CaDA](https://github.com/CIAM-Group/CaDA) | CaDA |
| Baseline | [ziweileonhuang/reld-nco](https://github.com/ziweileonhuang/reld-nco) | ReLD-MTL |
| Baseline | [naver/goal-co](https://github.com/naver/goal-co) | GOAL |
| Chuyên biệt | [yd-kwon/POMO](https://github.com/yd-kwon/POMO), [CIAM-Group/ICAM](https://github.com/CIAM-Group/ICAM), [naver/bq-nco](https://github.com/naver/bq-nco), [gaocrr/ELG](https://github.com/gaocrr/ELG) | Số tham khảo từng bài |
| Oracle | [PyVRP](https://github.com/PyVRP/PyVRP), [OR-Tools](https://github.com/google/or-tools), LKH-3 | Tính gap |

### 8.2 Cấu trúc gói

```
URS-fork/                     # fork của CIAM-Group/URS, giữ nguyên các thư mục gốc
  env/  problem/  data/  utils/  model/  ...   # KHÔNG sửa; chỉ import
  matroute/
    layout.py          # BlockLayout: tên khối, kích thước, chỉ số chiều, active_idx(S)
    spec.py            # ánh xạ tên biến thể -> tập khối S (dựa trên ProblemRep của URS)
    nn/
      block_linear.py  # trọng số dày + mặt nạ block-causal, chế độ lát cắt khi suy luận
      block_norm.py    # instance norm theo khối
      block_attn.py    # nhóm head theo khối, bias cặp (D, D^T, quan hệ PD)
      block_ffn.py
    encoder.py         # L lớp block-causal; bản attention thưa kNN cho N lớn
    decoder.py         # truy vấn theo khối, funnel hai tầng
    policy.py          # forward(S, level, k) -> logp, shortlist, logit từng khối
    features.py        # gói đặc trưng URS (UDR) thành các nhóm theo khối
    train/
      sampler.py       # lấy mẫu (task, mức) kiểu sandwich
      nested_reinforce.py   # L_RL + L_dist, chuẩn hóa reward theo biến thể
    eval/
      tester.py        # Tester mới, gọi policy MatRoute, dùng DataFinder/Reader của URS
    configs/           # d128.yaml, d192.yaml, p1.yaml, p2.yaml, ablation/*.yaml
    tests/
  train_matroute.py    # entry point huấn luyện
  test_matroute.py     # entry point đánh giá
```

### 8.3 Các điểm kỹ thuật cần làm đúng

1. **Mặt nạ trọng số thay cho lát cắt khi huấn luyện.** Dùng một ma trận dày nhân với mặt nạ cố định (buffer) trong huấn luyện cho đơn giản và nhanh trên GPU. Chỉ khi suy luận mới lát cắt để tiết kiệm FLOPs. Phải có kiểm thử chứng minh hai đường cho cùng đầu ra.
2. **Cô lập gradient của lõi.** Vì CORE không đọc khối ràng buộc, gradient từ khối ràng buộc vẫn chảy ngược vào CORE qua đường "khối đọc CORE". Nếu muốn lõi thật sự bất biến theo task, thử chế độ stop-gradient trong ablation (mặc định để gradient chảy).
3. **Khởi tạo khối mới bằng 0.** Đầu ra của khối ràng buộc khởi tạo gần 0 để lúc đầu mô hình giống TSP-solver rồi học dần ràng buộc.
4. **Shortlist theo khối phải tôn trọng mặt nạ khả thi.** Lọc node khả thi trước khi top-k. Nếu số node khả thi nhỏ hơn k thì lấy tất cả.
5. **Funnel khi huấn luyện ở N=100.** Với N=100 chỉ dùng k trong {16, 32, 64} để rerank thực sự chạy trên tập con.
6. **Hiệu quả.** Dùng bf16 cho encoder, giữ logit và softmax ở fp32. Dùng `torch.compile` sau khi đã có kiểm thử đúng. Cache khóa và giá trị của decoder như POMO.
7. **Tái lập.** Cố định seed dữ liệu test như URS, lưu cấu hình và băm commit của mã URS dùng để sinh dữ liệu.

### 8.4 Phác thảo mã cốt lõi

```python
class BlockLinear(nn.Module):
    # parents[b] = tập khối mà khối b được đọc; CORE chỉ đọc CORE
    def __init__(self, layout, parents):
        super().__init__()
        d = layout.dim
        self.weight = nn.Parameter(torch.empty(d, d))
        mask = torch.zeros(d, d)
        for b in layout.blocks:
            for p in parents[b]:
                mask[layout.idx(b)[:, None], layout.idx(p)[None, :]] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, h, active=None):
        w = self.weight * self.mask
        if active is not None and not self.training:   # suy luận: lát cắt
            idx = active
            return h[..., idx] @ w[idx][:, idx].T
        return h @ w.T
```

```python
def funnel_logits(q, k, active_blocks, topk, feasible):
    # tầng 0: điểm chiều thấp theo khối, chỉ trên node khả thi
    cand = torch.zeros_like(feasible)
    for b in active_blocks:
        s = (q[b][:, None, :] * k[b]).sum(-1) / math.sqrt(q[b].shape[-1])
        s = s.masked_fill(~feasible, float("-inf"))
        cand |= topk_mask(s, topk[b])
    # tầng 1: điểm đầy đủ, chỉ trên ứng viên
    u = sum((q[b][:, None, :] * k[b]).sum(-1) / math.sqrt(q[b].shape[-1])
            for b in active_blocks)
    return zeta * torch.tanh(u).masked_fill(~cand, float("-inf"))
```

```python
def train_step(batch, policy, sampler, stats):
    P, S = sampler.task()                 # biến thể và tập khối
    level, k = sampler.level()            # FULL | FUNNEL | CORE_ONLY
    logp, reward, shortlist_info = policy.rollout(batch, S, level, k)
    adv = (reward - reward.mean(1, keepdim=True)) / stats[P].std   # baseline POMO + chuẩn hóa theo biến thể
    loss_rl = -(adv.detach() * logp).mean()
    loss_dist = policy.distill_loss(batch, S) if level != "FULL" else policy.distill_loss(batch, S, from_full=True)
    return loss_rl + sampler.lambda_dist * loss_dist
```

Phác thảo trên chỉ minh họa luồng; tên hàm và chữ ký cần khớp khung huấn luyện đã chọn.

### 8.5 Kiểm thử đơn vị bắt buộc

- **Prefix-equivalence:** đầu ra CORE trên instance TSP không đổi khi các khối khác nhận đầu vào ngẫu nhiên.
- **Mask-slice:** lát cắt khi suy luận cho cùng logit với mặt nạ dày (sai số < 1e-5).
- **Funnel K=N:** logit funnel bằng softmax đầy đủ trên node khả thi.
- **Tính khả thi:** mọi lời giải sinh ra thỏa ràng buộc theo hàm kiểm tra của môi trường URS (tỷ lệ bất khả thi bằng 0).
- **Đối xứng/bất đối xứng:** với bài đối xứng, hai nửa head D và Dᵀ cho cùng điểm.
- **Tái tạo baseline:** chạy lại một baseline (ví dụ POMO trên CVRP100) và khớp số công bố trong ngưỡng seed trước khi tin mọi so sánh.

## 9. Rủi ro, kế hoạch dự phòng và go/no-go

Rủi ro lớn nhất là lõi chỉ-đọc quá hạn chế cho các bài có time window, và funnel loại nhầm node bắt buộc ở bài ràng buộc chặt. Cả hai đều có thí nghiệm kiểm chứng sớm và phương án dự phòng.

| Rủi ro | Dấu hiệu | Phương án dự phòng |
| --- | --- | --- |
| Lõi chỉ-đọc làm yếu bài TW/PD | Gap TW cao hơn bản phẳng trên E1 | Lõi cổng mềm: lớp trên cho CORE đọc khối ràng buộc với cổng khởi tạo 0; đưa vào thành thiết kế chính |
| Interference vẫn còn | Khoảng cách MTL–STL không nhỏ hơn URS | Tăng tỷ lệ chưng cất; thử stop-gradient từ khối ràng buộc về CORE; tăng kích thước CORE |
| Funnel loại nhầm node | Recall@K thấp ở TW chặt, gap tăng theo k nhỏ | Tăng k\_b của TW; thêm đề cử từ khối PRZ/DEM; cuối cùng dùng k thích ứng theo số node khả thi |
| Encoder O(N²) OOM | OOM ở N ≥ 5000 | Encoder kNN thưa; chia encoder theo khối (chỉ CORE chạy toàn đồ thị) |
| Lệch train/test do encoder thưa | Gap tăng ở N lớn so với bản đầy đủ ở N=1000 | Huấn luyện có encoder thưa trong một phần batch |
| MD, L, BP không có khối | Gap MD không tốt hơn URS | Chấp nhận trong claim; thêm khối RTE mở rộng và huấn luyện P2 có MD để xem trần |
| Chi phí RL nhân lên theo mức | Thời gian huấn luyện vượt 1.5 lần URS | Một mức mỗi batch (đã có), giảm tần suất CORE-ONLY |
| Chồng lấn novelty (slimmable, MatFormer, RouteFinder) | Reviewer coi là ghép kỹ thuật | Ablation Slim và Flat trong E8; nhấn mạnh E3 (tổ hợp) và funnel như phần mới |
| Baseline không tái lập được | RF NaN sau 150 epoch như URS báo | So ở 150 epoch như URS; ghi rõ trong bài |

### Tiêu chí go/no-go

Các ngưỡng dưới đây là đề xuất của mình, cần chốt cùng nhóm trước khi chạy.

| Mốc | Điều kiện đi tiếp | Nếu không đạt |
| --- | --- | --- |
| G1: sau khi có E1 bản d=128 (khoảng tuần 6) | Gap trung bình seen không tệ hơn Flat quá 0.3 điểm phần trăm; kiểm thử đơn vị đều qua | Chuyển sang lõi cổng mềm; nếu vẫn không đạt, bỏ khẳng định về lõi chỉ-đọc |
| G2: sau E3 (khoảng tuần 9) | MatRoute thấp hơn Flat về composition gap, chênh lệch lớn hơn độ lệch chuẩn 3 seed | Hạ trọng tâm bài xuống funnel và hiệu quả; giữ nested như thiết kế phụ |
| G3: sau E6 (khoảng tuần 11) | Ở N=2000: mất dưới 1% gap, speedup decoder từ 1.5 lần | Bỏ khẳng định tốc độ; giữ funnel như công cụ chất lượng hoặc bỏ hẳn |

**Mở rộng tùy chọn (chỉ làm nếu G1–G3 qua).** Thêm trục thứ hai: thang chiều lồng bên trong CORE (ví dụ 32/64/80 chiều) để đổi chất lượng lấy tốc độ theo kiểu MRL gốc, dùng cho chế độ anytime ở E7. Thử chọn mức theo độ khó instance.

## 10. Timeline, compute và công bố

Kế hoạch dài 16 tuần với ba cổng go/no-go (G1–G3) và tổng khoảng 2,000 GPU-giờ trên RTX 4090. Mọi con số dưới đây là **ước tính của mình**, dựa trên mốc 95 giờ cho một lần huấn luyện URS d=128 (paper URS), và cần hiệu chỉnh sau pilot. Giả định nhóm 2–3 người và 2–4 GPU; cần xác nhận.

### 10.1 Ngân sách compute (ước tính)

| Hạng mục | Số lần chạy | GPU-giờ mỗi lần | Tổng GPU-giờ |
| --- | --- | --- | --- |
| MatRoute d=192, P1, 3 seed | 3 | 140 | 420 |
| MatRoute d=128, P1, 3 seed (so khớp tham số) | 3 | 95 | 285 |
| MatRoute P2 (22 biến thể) | 1 | 160 | 160 |
| MatRoute-STL cho TSP, ATSP, CVRP, CVRPTW | 4 | 40 | 160 |
| Ablation E8 (lịch rút gọn 250 epoch) | 12 | 50 | 600 |
| Split tổ hợp E3 (MatRoute, Flat, URS; 2 split) | 6 | 60 | 360 |
| Huấn luyện lại RF-TE và CaDA (150 epoch) | 2 | 30 | 60 |
| **Tổng** |  |  | **2,045** |

Với 4 GPU chạy song song, tổng này khoảng 3 tuần liên tục; với 2 GPU khoảng 6 tuần. Phần oracle (PyVRP, OR-Tools) chạy trên CPU và ưu tiên dùng lại số của URS khi cùng dữ liệu để tiết kiệm.

**Phân chia local và server.** Máy local chỉ dùng để kiểm tra chức năng (kiểm thử đơn vị, smoke test N=20, overfit một batch, đánh giá 64 instance, vi kiểm tra tốc độ với trọng số ngẫu nhiên). Mọi con số trong bảng trên và các cổng G1–G3 chỉ được đánh giá sau khi chạy trên server.

### 10.2 Lịch 16 tuần

| Tuần | Công việc | Đầu ra |
| --- | --- | --- |
| 1–2 | Dựng môi trường, chạy lại URS và một baseline; viết adapter dữ liệu | Số baseline khớp paper |
| 3–4 | Cài đặt CORE và các khối, encoder block-causal, decoder funnel, kiểm thử đơn vị | Bộ kiểm thử qua hết |
| 5–6 | E1 bản d=128 và pilot nhỏ; cổng **G1** | Quyết định lõi chỉ-đọc hay cổng mềm |
| 7–9 | E8 nhóm cấu trúc, E2 và E3; cổng **G2** | Bảng zero-shot và composition gap |
| 10–11 | E4–E6: quy mô lớn, benchmark thật, phân tích funnel; cổng **G3** | Đường cong gap theo k, speedup |
| 12–13 | E7, E9, huấn luyện P2, chạy nốt seed | Phân tích biểu diễn, anytime |
| 14–16 | Viết bài, vẽ hình, kiểm tra tái lập, chuẩn bị mã công khai | Bản thảo hoàn chỉnh |

### 10.3 Phân công gợi ý

| Vai trò | Phạm vi |
| --- | --- |
| Mô hình | Khối, encoder, decoder funnel, kiểm thử đơn vị |
| Huấn luyện và baseline | Vòng RL, sampler, tái lập URS/RF/CaDA, quản lý GPU |
| Thực nghiệm và viết | E1–E9, bảng, hình, viết bài |

### 10.4 Công bố và đạo đức nghiên cứu

- Chọn hội nghị mục tiêu sau khi có kết quả G2; mốc hạn nộp cần xác nhận lại ngay trước khi lập kế hoạch, mình chưa kiểm tra.
- Phát hành mã, cấu hình và checkpoint. Repo URS dùng giấy phép AGPLv3, nên mã dẫn xuất khi phân phối hoặc chạy như dịch vụ mạng phải công khai dưới cùng giấy phép (cần xác nhận pháp lý nếu muốn giấy phép khác); ghi rõ commit của mã URS dùng để sinh dữ liệu và giấy phép các repo (một số repo như BQ, LEHD, GOAL chỉ cho phép phi thương mại theo paper URS).
- Trích dẫn đầy đủ URS, RouteFinder, MVMoE, CaDA, MRL, MatFormer, slimmable networks; nêu rõ phần nào là giao diện dữ liệu mượn từ benchmark.
- Trước khi submit, quét lại arXiv với từ khóa Matryoshka/nested/prefix kết hợp routing để chắc không có công trình trùng ý tưởng.
