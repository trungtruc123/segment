# Fine-tune data_v1 → data_v2

Tài liệu này ghi lại (1) sự khác nhau giữa `data_v1` và `data_v2` cùng ảnh hưởng
tới chất lượng training, và (2) cách chạy pipeline fine-tune mới.

---

## 1. So sánh data_v1 vs data_v2

Số liệu đo trực tiếp từ file NIfTI (header + toàn bộ voxel của mask).

| | **data_v1** | **data_v2** |
|---|---|---|
| Số ca CBCT | 5 (SLZ000–SLZ004) | 9 (LD1_LD2, LD3_LD4, LD5_NT11, LT2_NT2, LT4_NT4, LT5_NT5, LT6_NT6, LT7_NT7, L10_NT10) |
| Shape | 503×503×501 (SLZ004: 512×512×384) | 714×714×714 — **đồng nhất** |
| Spacing | 0.08 mm (SLZ004: **0.16 mm**) | 0.07 mm — **đồng nhất** |
| FOV | 40×40×40 mm (SLZ004: 82×82×61 mm) | 50×50×50 mm |
| dtype ảnh | `int16` | **`float64`** |
| dtype nhãn | `int16` | `int16` |
| Orientation | LPS | LPS |
| Giá trị nhãn | {0, 1, 2} | {0, 1, 2} — **giống hệt** |
| Dải intensity | [-3140, 12380] (SLZ004: [-790, 3071]) | [-1456, 6230] |
| Nền (background) | phân tán, không có giá trị cố định | **hằng số -1456** chiếm >50% volume |
| Răng / ca | ~6 (tổng 25 răng) | ~2 (tổng ~18 răng) |
| Voxel tooth / ca | ~6.3 M (SLZ004: 0.23 M) | 3.8 – 5.4 M |
| Voxel canal / ca | 0.37 – 0.49 M (SLZ004: 0.01 M) | 0.15 – 0.36 M |
| Tỉ lệ canal/tooth | 5.7 – 7.7 % | 3.4 – 6.8 % |
| Thể tích / răng | ~540 mm³ | ~700–800 mm³ |

### Điểm giống nhau quan trọng nhất

- **Quy ước nhãn hoàn toàn giống** (0=bg, 1=tooth, 2=canal) → không cần remap.
- **Orientation giống** (LPS) → `Orientationd(axcodes="RAS")` xử lý được cả hai.
- **Mất cân bằng lớp tương đương** (canal ~5% thể tích răng) → giữ nguyên
  `class_weights=[1,1,5]` và `canal_oversample_ratio=3.0` là hợp lý.
- **Cùng loại giải phẫu, cùng thang mm** → sau `Spacingd(0.3mm)` thì kích thước
  vật lý của patch 96³ (28.8 mm) vẫn bao trọn một răng ở cả hai tập.

---

## 2. Khác biệt nào ẢNH HƯỞNG tới chất lượng training?

### 2.1 Gần như KHÔNG ảnh hưởng — pipeline hiện tại đã xử lý

| Khác biệt | Vì sao không đáng lo |
|---|---|
| Spacing 0.08 → 0.07 mm | `Spacingd(pixdim=(0.3,0.3,0.3))` resample cả hai về cùng lưới trước khi vào model. Model không bao giờ "nhìn thấy" spacing gốc. |
| Shape 503³ → 714³ | Training chạy trên **răng đã cắt rời** rồi crop patch 96³, không phải trên volume gốc. |
| Dải intensity khác hẳn | `ScaleIntensityRangePercentilesd(0.5, 99.5)` chuẩn hoá **theo phân vị của từng volume** về [0,1] — đúng cách làm cho CBCT (không có HU chuẩn). Đây là thứ cứu cả pipeline; nếu trước đây dùng `ScaleIntensityRange` với min/max cố định thì data_v2 đã hỏng hoàn toàn. |
| dtype `float64` vs `int16` | Không ảnh hưởng độ chính xác (đã bàn về RAM ở dưới). |

### 2.2 CÓ ảnh hưởng — cần chú ý

**(a) RAM khi split — nghiêm trọng nhất, sẽ crash nếu không sửa**

714³ = 364 triệu voxel. `nib.get_fdata()` luôn trả `float64` → **2.9 GB cho một
volume**. Code cũ giữ đồng thời: image float64 (2.9 GB) + label float64 (2.9 GB)
+ `label.astype(int)` int64 (2.9 GB) + components int32 (1.5 GB) + bản relabel
(1.5 GB) ≈ **12 GB** → OOM trên Colab standard (12.7 GB). Ngoài ra vòng lặp
relabel cũ tạo một boolean mask full-volume cho **mỗi** component (data_v1 có ca
tới 54 component) → cực chậm ở 714³.

→ Đã sửa trong `split_teeth.py`: đọc qua `dataobj` và ép `image→float32`,
`label→uint8`, relabel bằng lookup-table, `del` các mảng trung gian. Peak RAM còn
khoảng 4 GB.

**(b) Số ca nhiều hơn nhưng ít răng hơn — đây là tin TỐT**

data_v2 cho **9 ca** thay vì 5. Với segmentation y tế, biến thiên giữa **bệnh
nhân/máy chụp** mới là nguồn khó khái quát hoá, chứ không phải số răng — 6 răng
trong cùng một CBCT gần như là 6 mẫu tương quan cao (cùng nhiễu, cùng artifact,
cùng phơi nhiễm). 9 ca độc lập của v2 làm tăng đáng kể độ đa dạng domain, dù chỉ
thêm ~18 răng. Dự kiến model sau fine-tune **khái quát hoá tốt hơn** model v1
hiện tại, kể cả khi val dice tuyệt đối thấp hơn 0.96.

**(c) Nền hằng số -1456 làm `CropForegroundd` hoạt động khác nhau**

Trong `get_train_transforms`, `CropForegroundd(source_key="image")` chạy **sau**
bước chuẩn hoá và dùng mặc định `is_positive` (giữ voxel > 0):

- data_v2: nền = -1456 = đúng phân vị 0.5% → sau clip+scale thành **0.0** → bị
  crop bỏ, ảnh cắt sát vào vùng có mô.
- data_v1: nền ≈ -1605 trong khi phân vị 0.5% là -3140 → thành ~0.16 > 0 → **hầu
  như không crop gì**.

Nghĩa là lượng "khí/nền" quanh mỗi răng khác nhau giữa hai tập. Ảnh hưởng ở mức
nhẹ (crop răng đơn đã bám sát răng, `margin=15` voxel), nhưng nếu sau này thấy
model nhạy với viền, hãy đổi sang ngưỡng tường minh:

```python
CropForegroundd(keys=keys, source_key="image", margin=20,
                select_fn=lambda x: x > 0.05)
```

**(d) SLZ004 là ca lệch chuẩn của chính data_v1**

Spacing 0.16 mm (gấp đôi), FOV 82 mm, chỉ 1 răng, canal chỉ 9975 voxel — trong
lần train trước fold 2 (val = SLZ004) là fold tệ nhất: dice 0.9076 so với
0.93–0.96 của các fold khác. Ca này vẫn giữ trong train, nhưng đừng dùng nó làm
thước đo.

**(e) Val dice sẽ TỤT so với 0.9605 — và đó là bình thường**

0.9605 của fold1 là dice trên SLZ001, tức một răng **cùng ca chụp, cùng máy**
với tập train. Dice trên ca v2 hoàn toàn mới sẽ thấp hơn; con số ~0.88–0.93 vẫn
là kết quả tốt hơn về mặt thực dụng. Đừng so trực tiếp hai con số.

### 2.3 Kết luận ngắn

> Khác biệt giữa hai tập **không đủ lớn để phá hỏng việc training**: cùng quy ước
> nhãn, cùng orientation, cùng mức mất cân bằng lớp, và pipeline đã chuẩn hoá cả
> spacing lẫn intensity theo từng volume. Rào cản thật sự là **RAM ở bước split**
> (đã sửa) chứ không phải chất lượng học. Điều cần điều chỉnh là **cách đánh giá**,
> không phải kiến trúc hay loss.

---

## 3. Những gì đã thay đổi trong code

| File | Thay đổi |
|---|---|
| `split_teeth.py` | `load_nifti(path, dtype)` đọc qua `dataobj` thay cho `get_fdata()` (tránh float64); `process_case` load image→float32, label→uint8; `find_individual_teeth` relabel bằng lookup-table + `del` mảng trung gian. |
| `dataset.py` | `prepare_data_list(dir, source=...)` gắn thêm khoá `"source"`; thêm `prepare_multi_data_list()` gộp nhiều thư mục teeth (tự prefix khi trùng case_id); thêm `oversample_by_source()`; `kfold_split_by_case()` thêm `val_sources=` và `always_train_sources=`. |
| `finetune.py` | **Mới.** `FineTuneTrainer` + K-fold có auto-resume, nạp pretrained weights, freeze/unfreeze encoder, báo cáo dice tách riêng theo domain. |
| `training.ipynb` | Cấu hình lại đường dẫn cho v1/v2, split cả hai tập, thay phần training bằng lời gọi `finetune.run_finetune()`; bỏ đoạn vá `losses.py` (repo đã dùng `register_buffer` từ lâu); cell 6a tự chọn preset theo GPU. |
| `train.py` | `autocast` nhận `dtype` từ `self.amp_dtype` → hỗ trợ bf16 trên Ampere. |
| `dataset.py` (dataloader) | `get_fold_dataloaders` thêm `persistent_workers` + `prefetch_factor` — dataset nhỏ nên chi phí fork worker mỗi epoch đáng kể khi GPU nhanh. |

---

## 4. Cách chạy

### Colab
Chạy tuần tự `training.ipynb`. Cell **6a** in ra cách chia fold để kiểm tra trước
khi tốn GPU; cell **6b** chạy fine-tune. Colab ngắt thì chạy lại 6b — tự resume.

### CLI

```bash
# Full fine-tune (khuyến nghị)
python finetune.py \
    --teeth_dir_v1 data_v1/teeth \
    --teeth_dir_v2 data_v2/teeth \
    --pretrained checkpoints_v1/kfold_nnunet_tooth_canal/fold1/best_model.pth \
    --checkpoint_dir checkpoints_v2 --experiment kfold_ft_v1v2 \
    --n_folds 5 --epochs 100 --lr 1e-4

# Freeze encoder 20 epoch đầu rồi mở toàn bộ với LR thấp
python finetune.py ... --freeze_encoder_epochs 20 --head_lr 3e-4 --lr 5e-5

# Chỉ in cách chia fold, không cần GPU
python finetune.py ... --dry_run
```

### Các flag đáng chú ý

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--val_sources` | `v2` | Chỉ validate trên ca của data_v2; toàn bộ v1 bị **ghim vào train ở mọi fold**. Vì checkpoint xuất phát đã học 4/5 ca v1, để ca v1 vào val sẽ cho dice ảo cao. Dùng `all` nếu vẫn muốn validate cả hai. |
| `--freeze_encoder_epochs` | `0` | `0` = full fine-tune. `>0` = giai đoạn 1 chỉ train decoder + head với `--head_lr`, sau đó mở toàn mạng với `--lr` và cosine trên số epoch còn lại. |
| `--lr` | `1e-4` | Bằng 1/3 LR của lần train gốc (3e-4). Fine-tune với LR gốc dễ xoá sạch feature đã học. |
| `--v2_repeat` | `1` | Nhân bản mỗi răng v2 trong train N lần. Đặt `2` nếu v2 mới là domain đích thực sự (train sẽ thành 30 răng v1 + 36 lượt v2). |
| `--epochs` | `100` | Fine-tune hội tụ nhanh hơn train from scratch (gốc 200). |
| `--only_fold` | – | Chạy lại đúng một fold. |
| `--batch_size` | `4` | **Số ITEM, không phải số patch.** `RandCropByPosNegLabeld(num_samples=2)` cho 2 patch mỗi item → patch/step = `batch_size × 2`. |
| `--amp_dtype` | `fp16` | `bf16` chỉ chạy từ Ampere (A100/L4/H100). Nếu GPU không hỗ trợ, `run_finetune` tự hạ về fp16. |
| `--sw_batch_size` | `4` | Số patch mỗi lượt sliding-window khi validate. |

---

## 4b. Chạy trên GPU mạnh (A100)

`finetune.setup_gpu()` được gọi tự động ở đầu `run_finetune()` và bật:

- **TF32** (`torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32`) — Ampere
  chạy conv/matmul fp32 trên tensor core, nhanh hơn đáng kể, không đổi chất lượng
  segmentation. Không có tác dụng trên T4 (Turing).
- **`cudnn.benchmark = True`** — cuDNN tự chọn thuật toán conv nhanh nhất. Chỉ có
  lợi khi shape input cố định, đúng với pipeline này (patch 96³ khi train, roi 96³
  khi sliding-window val).

### Preset theo GPU

| GPU | `--batch_size` | patch/step | `--num_workers` | `--amp_dtype` | `--sw_batch_size` | `--lr` |
|---|---|---|---|---|---|---|
| T4 16GB | 4 | 8 | 2 | `fp16` | 4 | 1e-4 |
| L4 / A10 24GB | 6 | 12 | 4 | `bf16` | 6 | 1.2e-4 |
| **A100 40GB** | **8** | **16** | **6** | **`bf16`** | **8** | **1.4e-4** |

Notebook (cell 6a) tự chọn preset theo VRAM đọc được, nên không phải sửa tay.

### Vì sao bf16 thay vì fp16 trên A100

bf16 có dynamic range bằng fp32 nên **không cần `GradScaler`** và không bao giờ
overflow thành `inf`. Điều này quan trọng với pipeline hiện tại vì deep supervision
cộng 3 loss ở 3 scale, mỗi loss lại là Dice + Focal có `class_weights=[1,1,5]` —
fp16 dễ underflow gradient của class canal (class nhỏ nhất). `FineTuneTrainer` tự
tắt `GradScaler` khi dùng bf16.

### Vì sao scale LR khi tăng batch

Batch tăng gấp đôi (8 → 16 patch/step) làm gradient noise giảm, nên LR nên tăng
theo. Ở đây dùng quy tắc **căn bậc hai** (`1e-4 × √2 ≈ 1.4e-4`) chứ không phải
linear scaling, vì đây là fine-tune — mục tiêu là đi chậm để không phá weights đã
học, không phải hội tụ nhanh nhất.

### Điều A100 KHÔNG giải quyết

- **Bước split** (`split_teeth.py`) chạy hoàn toàn trên CPU + RAM, không dùng GPU.
  Đây vẫn là bước nặng nhất về bộ nhớ với data_v2 (714³).
- **Dataset quá nhỏ** (~43 răng) nên mỗi epoch chỉ vài chục step. GPU nhanh sẽ dễ
  bị *starve* vì dataloader — đó là lý do bật `persistent_workers=True` và
  `prefetch_factor=4` trong `get_fold_dataloaders`. Nếu vẫn thấy GPU util thấp,
  tăng `--num_workers` trước khi tăng `--batch_size`.
- **Không nên tăng `--epochs`** chỉ vì train nhanh hơn. Với 43 mẫu, `early_stopping_patience=25`
  thường dừng trước 100 epoch; train lâu hơn chỉ overfit.

### Chia fold thực tế (n_folds=5, seed=42)

```
5 ca v1 luôn nằm trong train: SLZ000..SLZ004
9 ca v2 chia đều làm validation:
  Fold 1: val = LD3_LD4, LT6_NT6
  Fold 2: val = LD1_LD2, LT2_NT2
  Fold 3: val = LD5_NT11, LT4_NT4
  Fold 4: val = L10_NT10, LT5_NT5
  Fold 5: val = LT7_NT7
```

---

## 5. Đọc kết quả thế nào

`checkpoints_v2/kfold_ft_v1v2/kfold_summary.json` ghi `mean_dice ± std_dice` trên
các ca **v2 chưa từng thấy**. So sánh đúng cách:

- **Không** so với `mean_dice = 0.9421` của lần train v1 (khác tập val, khác domain).
- Muốn biết fine-tune có lợi hay không, hãy chạy baseline: dùng thẳng checkpoint
  v1 (chưa fine-tune) để inference trên các ca v2 và đo dice. Chênh lệch giữa hai
  con số mới là mức cải thiện thật.
- Nhật ký training in `dice_mean_v1` / `dice_mean_v2` riêng khi val có cả hai
  domain (`--val_sources all`) — dùng để phát hiện catastrophic forgetting.
