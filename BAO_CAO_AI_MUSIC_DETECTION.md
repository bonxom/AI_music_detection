# Báo cáo dự án AI-generated Audio Detection

## 1. Giới thiệu bài toán

Mục tiêu của dự án là phân loại một đoạn âm thanh hoặc phần âm thanh trích từ video thành một trong hai lớp:

- **Synthetic/AI-generated** (`label = 0`): nhạc hoặc giọng nói được sinh/tổng hợp bởi AI.
- **Real/Human** (`label = 1`): âm thanh có nguồn gốc tự nhiên/do con người tạo ra.

Đầu vào triển khai là tệp audio hoặc video. Hệ thống chuẩn hoá tín hiệu về mono, 16 kHz và suy luận trên các đoạn 6 giây. Đầu ra gồm nhãn dự đoán, xác suất AI và thông tin đoạn âm thanh đáng ngờ nhất. Bài toán khó ở chỗ chất lượng audio, nhạc nền, ngôn ngữ, giọng hát và công cụ sinh âm thanh rất đa dạng; vì vậy điểm số cao trên tập chia ngẫu nhiên không tự động phản ánh khả năng tổng quát trên nguồn hoặc công cụ AI chưa từng thấy.

## 2. Dữ liệu và tiền xử lý dữ liệu

### 2.1. Tổ chức dữ liệu và nhãn

Các CSV đã có trong `crop_data/` mô tả mỗi mẫu bằng `filename`, `path`, `fake_type` và `label`. Số lượng tại thời điểm lập báo cáo:

| Tập | Real (1) | Synthetic (0) | Tổng |
| --- | ---: | ---: | ---: |
| `crop_data/crop6.csv` | 28.112 | 42.370 | 70.482 |
| `crop_data/test.csv` | 20.000 | 7.464 | 27.464 |

Ngoài các CSV nhạc hiện có trong `archive/`, `data_process/audio_datasets.yaml` là catalog các nguồn Hugging Face cho audio thật và synthetic/deepfake. Các nguồn này bao gồm Common Voice, VoxPopuli, GigaSpeech, ASVspoof, ElevenLabs, ShiftySpeech và nhiều corpus đa ngôn ngữ khác. Hàm `build_binary_audio_df_from_yaml()` trong `data_process/hf_audio_df.py` tải từng dataset, tự nhận diện cột audio (hoặc dùng cột chỉ định), lưu clip ra thư mục cục bộ, gán nhãn `real = 1`, `synthetic = 0`, rồi chia train/test theo stratified split với `test_size=0.2` và `random_state=42`.

### 2.2. Nhánh đầu vào mel-spectrogram

`model_mel_input/dataset.py` tải file bằng Librosa, chuyển mono, resample về 16 kHz và tạo mel-spectrogram với các tham số:

- `n_fft = 2048`, `hop_length = 512`, `n_mels = 128`;
- dải tần 20–8.000 Hz;
- power-to-dB, sau đó chuẩn hoá z-score theo từng spectrogram.

Tensor đưa vào mô hình có dạng `[B, 1, 128, T]`; mô hình tự bỏ chiều channel thừa để làm việc với `[B, F, T]`.

### 2.3. Nhánh đầu vào waveform

`model_audio_input/dataset.py` đọc audio bằng SoundFile, trung bình các kênh để có mono và resample bằng `scipy.signal.resample_poly`. Tín hiệu được cắt hoặc zero-pad về đúng 96.000 samples, tương đương 6 giây ở 16 kHz. Dataset trả về waveform `[B, 96000]`; việc tạo mel-spectrogram nằm bên trong `forward()` của model. Cách tổ chức này đảm bảo tiền xử lý train/inference có thể đồng nhất khi đóng gói mô hình.

Trong backend, audio tải lên được đọc bằng Librosa. Nếu input là video, hệ thống dùng `ffmpeg` tách audio tạm thời rồi áp dụng cùng pipeline. Chế độ `full_crop` chia toàn bộ waveform thành các cửa sổ liên tiếp 6 giây và zero-pad cửa sổ cuối. Chế độ có tên `random_6s_crop` hiện thực tế lấy **6 giây đầu** của file, không lấy ngẫu nhiên; đây là điểm cần đổi tên hoặc sửa implementation để tránh sai lệch giữa giao diện và hành vi.

## 3. Mô hình đề xuất

### 3.1. SpecTTTra: Spectro-Temporal Tokens Transformer

Hai nhánh cùng sử dụng ý tưởng **SpecTTTra**: thay vì chia mặt phẳng spectrogram thành patch 2D như ViT, mô hình tạo token độc lập theo hai hướng.

1. **Temporal tokenizer**: Conv1D quét theo trục thời gian, với toàn bộ `F` mel bins là channel đầu vào.
2. **Spectral tokenizer**: hoán vị spectrogram rồi Conv1D quét theo trục tần số, với toàn bộ `T` frame là channel đầu vào.
3. Mỗi nhánh dùng GELU và positional encoding sinusoidal không học; token của hai nhánh được nối lại.
4. Chuỗi token đi qua Transformer encoder, rồi adaptive average pooling và tầng tuyến tính phân lớp.

Với `F` mel bins, `T` time frames, kích thước spectral clip `f` và temporal clip `t`, số token xấp xỉ:

`N = floor(F / f) + floor(T / t)`

Thay vì tăng theo tích của hai chiều như token patch 2D, công thức trên tăng gần tuyến tính theo độ dài audio. Đây là lựa chọn phù hợp hơn khi quét bài nhạc dài bằng nhiều cửa sổ.

`layer/tokenizer.py` hiện thực hai tokenizer Conv1D và `layer/pos_encoding.py` hiện thực positional encoding sinusoidal dưới dạng buffer, nên không phát sinh tham số học cho vị trí.

### 3.2. Biến thể mel-spectrogram

`model_mel_input/spectttra.py` nhận mel đã xử lý trước. Transformer encoder dùng `embed_dim`, `num_heads`, `num_layers` truyền từ cấu hình, average pooling trên chiều token và classifier 1-logit. Biến thể này phù hợp cho thử nghiệm đặc trưng time–frequency tách rời khỏi model, nhưng cần bảo đảm code huấn luyện dùng loss nhị phân đúng với output một logit.

### 3.3. Biến thể waveform — mô hình được triển khai

`model_audio_input/model.py` nhận waveform và sinh MelSpectrogram trong model với cùng dải 20–8.000 Hz, `n_fft=2048`, `hop_length=512`, 128 mel bins, sau đó chuyển dB. Model căn chỉnh số frame về cấu hình để bảo đảm số token cố định.

Cấu hình mặc định `alpha` trong `config/model_hparams.yaml`, cũng là cấu hình mà backend chọn mặc định, là:

| Thuộc tính | Giá trị |
| --- | ---: |
| Thời lượng / sample rate | 6 giây / 16 kHz (96.000 samples) |
| Mel bins / time frames | 128 / 188 |
| Embedding dimension | 768 |
| `f_clip` / `t_clip` | 1 / 3 |
| Số token | 128 + `floor(188/3)` = 190 |
| Attention heads / encoder layers | 6 / 12 |
| Feed-forward dimension | 3.072 |
| Dropout | 0,2 |

Notebook `main_audio.ipynb` huấn luyện biến thể xác suất một logit bằng `BCEWithLogitsLoss`, AdamW (`lr=1e-5`, `weight_decay=1e-4`) và gradient clipping 1,0. Logit sau sigmoid là xác suất lớp real; backend nhất quán chuyển thành `P(AI) = 1 - sigmoid(logit)`. File trọng số triển khai là `app/model_prob/model_alpha.safetensors` (khoảng 402 MB). Mã cũng hỗ trợ checkpoint hai lớp: softmax với thứ tự `[AI, human]`.

## 4. Kết quả triển khai và nhận xét

### 4.1. Kết quả huấn luyện đã được lưu

`main_audio.ipynb` lưu một lần chạy 10 epoch trên **12.235 clip 6 giây**, chia stratified 80/20 thành 9.788 train và 2.447 test. Kết quả tốt nhất được ghi ở epoch 9:

| Chỉ số trên test split nội bộ | Giá trị |
| --- | ---: |
| Accuracy | 99,67% |
| Balanced accuracy | 99,70% |
| Recall synthetic | 99,50% |
| Recall real | 99,90% |
| Ma trận nhầm lẫn `[[synthetic→synthetic, synthetic→real], [real→synthetic, real→real]]` | `[[1393, 7], [1, 1046]]` |

Epoch cuối đạt accuracy 99,63% và balanced accuracy 99,63%. Đây là kết quả khả quan cho split nội bộ, nhưng chưa có output được lưu cho cell đánh giá riêng trên `crop_data/test.csv`; vì vậy không nên diễn giải các chỉ số này là khả năng tổng quát cuối cùng. Đặc biệt cần tách train/validation/test theo bài hát, nguồn dataset hoặc generator (không chỉ theo clip) để tránh các đoạn crop gần giống của cùng bản ghi xuất hiện ở cả hai phía của split.

### 4.2. Triển khai ứng dụng

Backend FastAPI cung cấp các endpoint:

- `GET /ai/models`: liệt kê checkpoint `.safetensors` trong `app/model_prob/`.
- `POST /ai/models/select`: nạp/chọn checkpoint; model đã nạp được cache.
- `POST /ai/detect/random-crop`: suy luận một đoạn 6 giây.
- `POST /ai/detect/full-crop`: suy luận theo batch trên mọi cửa sổ 6 giây liên tiếp.

Ở `full_crop`, backend chọn cửa sổ có xác suất non-AI nhỏ nhất (tương đương xác suất AI cao nhất), trả về xác suất của cửa sổ này, vị trí thời gian, số cửa sổ và danh sách xác suất non-AI. Ngưỡng quyết định là 0,5. Frontend React/Vite cho phép chọn model, chọn hai chế độ phát hiện, tải audio/video và hiển thị xác suất AI, nhãn, model sử dụng cùng vị trí đoạn nghi ngờ.

### 4.3. Nhận xét và hướng hoàn thiện

- Pipeline đã khép kín từ dữ liệu, training, đóng gói `safetensors` đến API và UI; đường triển khai đang dùng nhánh **waveform input**, không phải nhánh mel-input rời.
- Cần đánh giá checkpoint đang deploy trên `crop_data/test.csv` hoặc một holdout thật sự độc lập và báo cáo thêm precision, F1, ROC-AUC/EER, calibration cùng kết quả theo từng nguồn/generator.
- `crop6.csv` đang lệch lớp (synthetic nhiều hơn real). `main_audio.py` có weighted sampler và class-weighted CrossEntropy cho output hai lớp, trong khi notebook probability dùng BCE không nêu `pos_weight`; cần thống nhất biến thể model, loss, chiến lược cân bằng và checkpoint trước khi so sánh kết quả.
- Cần chuẩn hoá chính sách cắt đoạn: đổi `random_6s_crop` thành `first_6s_crop`, hoặc thực sự chọn vị trí ngẫu nhiên có seed; với full-crop, nên cân nhắc thêm một luật aggregate (tỷ lệ cửa sổ nghi ngờ hoặc top-k mean) vì chỉ lấy cực đại có thể nhạy với một cửa sổ nhiễu.
- Để bảo đảm tái lập, nên version hoá manifest dữ liệu, seed, checkpoint gắn với cấu hình, và lưu kết quả đánh giá độc lập cùng commit phát hành.
