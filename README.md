# Gold Miner - Reinforcement Learning Training

Dự án Gold Miner với AI agent được train bằng Deep Q-Learning.

## Cài đặt

```bash
pip install pygame torch gymnasium numpy einops tqdm
```

## Cách sử dụng

### 1. Chơi game thủ công

```bash
python runner/main.py
```

Chạy game Gold Miner bình thường, điều khiển bằng tay.

### 2. Training AI Agent

#### Train Qtention model (khuyến nghị)

```bash
python runner/train_qtention.py
```

**Các tham số có thể điều chỉnh:**

```bash
python runner/train_qtention.py \
  --n-cycles 2000 \              # Số vòng training
  --m-episodes 2 \               # Số episodes thu thập mỗi cycle
  --n-updates 3 \                # Số lần update Q-network mỗi cycle
  --batch-size 128 \             # Batch size
  --lr 3e-4 \                    # Learning rate
  --gamma 0.9 \                  # Discount factor
  --epsilon-start 0.9 \          # Epsilon bắt đầu
  --epsilon-end 0.5 \            # Epsilon kết thúc
  --epsilon-decay 0.995 \        # Tốc độ giảm epsilon
  --buffer-size 20000 \          # Kích thước replay buffer
  --target-update-freq 100 \     # Update target network mỗi N cycles
  --save-dir checkpoints/qtention \
  --show                         # Hiển thị game khi train (chậm hơn)
```

**Ví dụ train nhanh:**

```bash
python runner/train_qtention.py --n-cycles 1000 --game-speed 5
```

### 3. Đánh giá model (Evaluation)

```bash
python runner/eval.py --checkpoint checkpoints/qtention/checkpoint_cycle_1200.pth
```

**Các tham số:**

```bash
python runner/eval.py \
  --checkpoint <đường_dẫn_checkpoint> \  # Path to model checkpoint
  --num-episodes 5 \                     # Số episodes để test
  --fps 60 \                             # Frames per second
  --net attention \                      # Loại network: attention/cnn/rnn
  --seed 42                              # Random seed (optional)
```

**Ví dụ eval các models:**

```bash
# Qtention model
python runner/eval.py --checkpoint checkpoints/qtention/checkpoint_cycle_1200.pth --net attention

# QCNN model
python runner/eval.py --checkpoint checkpoints/goat.pt --net cnn

# QCnnRnn model
python runner/eval.py --checkpoint checkpoints/cnn_rnn_ckpt.pt --net cnn_rnn
```

### 4. Tạo warmup buffer (Optional)

Tạo replay buffer với random policy trước khi train:

```bash
python warmup_buffer_builders/warmup_buffer_qtention.py
```

File buffer sẽ được lưu vào `buffers/warmup_buffer_qtention.pkl` và tự động được load khi train.

## Checkpoints

Các pretrained models trong thư mục `checkpoints/`:

```
checkpoints/
├── qtention/
│   ├── checkpoint_cycle_1200.pth    # Qtention model (best)
│   ├── checkpoint_cycle_1000.pth
│   └── ...
├── goat.pt                          # QCNN model
├── cnn_rnn_ckpt.pt                  # QCnnRnn model
└── final_model_save.pt              # Final trained model
```

## Cấu trúc project

- `runner/` - Scripts chạy chính (main, train, eval)
- `model/` - Gymnasium environments
- `agent/` - Neural network architectures (QCNN, QCnnRnn, Qtention)
- `trainer/` - Training algorithms (DQN, Double DQN)
- `scenes/` - Game logic và rendering
- `entities/` - Game entities (miner, rope, gold...)
- `checkpoints/` - Saved models
- `buffers/` - Replay buffers
- `levels/` - Level definitions

## Tips

- Train với `--show` để xem agent học, nhưng sẽ chậm hơn nhiều
- Tăng `--game-speed` để train nhanh hơn (không hiển thị game)
- Checkpoint tự động save mỗi 10 cycles
- Xem training progress trong file log: `training_qtention.log`
- Qtention model cho kết quả tốt nhất nhưng train lâu hơn
