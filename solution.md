
# Mô hình hóa bài toán Gold Miner với Gymnasium + Pygame

## 1. Objective & MDP

- **Mục tiêu (objective)**:  
  Tối đa hóa **tổng điểm** trong một khoảng thời gian/num-steps giới hạn:

  \[
  \max_\pi \; \mathbb{E}_\pi\left[\sum_{t=0}^{T-1} r_t\right]
  \]

- Đây là **bài toán episodic**:
  - Mỗi episode có độ dài đúng 60s.
  - Kết thúc khi:
    - Hết thời gian (`time_left <= 0`) → `terminated = True`

- Giả thiết **Markov**:  
  State hiện tại chứa đủ thông tin để quyết định phân phối state kế và reward → bài toán thỏa **Markov property**.

---

## 2. Action Space

Ở mỗi thời điểm có **2 hành động**:

- `0`: **Không làm gì** (do nothing).
- `1`:  
  - Nếu **móc đang rảnh** → bắn/kéo móc.  
  - Nếu **móc đang kéo vật phẩm** → sử dụng dynamite (nếu còn dynamite).

## 3. Reward

- Deliver: `+V(item)`
- Dynamite: `-C_dyna` (và hủy item)
- Các bước khác (bắn trúng nhưng chưa giao / chờ dây thu): `0`
- (Tuỳ) Khi hook đang bận (đang kéo): −c_pull * g(weight)
- (Tuỳ) Step cost rất nhỏ `-c_step` để tránh đứng chờ vô hạn


## 4. Tokenization features

- **ENV token** (ví dụ):
  - `time_left`, `dynamite_count`, `rope_length_norm`, `rope_dir(sin,cos)`, `rope_has_item`, `weight_current_norm`.
- **ITEM token** (mỗi object một token):
  - **Type embedding**: Gold/Rock/Bag/Mole/Other.
  - **Hình học tương đối** so với móc: \[dx, dy, r, sin(\phi), cos(\phi)\] (chuẩn hóa theo kích thước map).
  - **Thuộc tính**: `value_norm`, `weight_norm`, `is_move`, `is_explosive`, `size`, `point`.
  - **Thuộc tính di chuyển**: `direction`, `range(l, r)`
  - *(Hữu ích)* đặc trưng dẫn xuất: **ước lượng thời gian kéo** \[ \hat T_i \approx \text{distance}_i / v_{\text{pull}}(weight) \].
  - **Pulling vector**: giúp agent biết được item nào đang được kéo
  