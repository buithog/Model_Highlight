from typing import List, Tuple
import csv
import numpy as np
import os

# Đặt cấu hình GPU nếu cần
os.environ["CUDA_VISIBLE_DEVICES"] = ''


# Hàm chuyển đổi thời gian từ dạng mm:ss sang giây
def convert_to_seconds(time_str: str) -> int:
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + int(seconds)
    return total_seconds


# Hàm kiểm tra overlap giữa 2 khoảng thời gian
def overlap(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> int:
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start)


# Tính precision, recall, f1-score
def calculate_metrics(predictions: List[Tuple[int, int]], true_labels: List[Tuple[int, int]]) -> Tuple[float, float, float]:
    # print(predictions)
    # print(true_labels)
    indexPrediction = 0
    indexLabel = 0
    maxOverlap = 0
    for i in range(len(predictions)):
        for j in range(len(true_labels)):
            if overlap(predictions[i], true_labels[j]) > maxOverlap:
                maxOverlap = overlap(predictions[i], true_labels[j])
                indexPrediction = i
                indexLabel = j

    precision = maxOverlap / (predictions[indexPrediction][1] - predictions[indexPrediction][0])
    recall = maxOverlap / (true_labels[indexLabel][1] - true_labels[indexLabel][0])
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# Tính spectral centroid
def calculate_spectral_centroid(spectrogram, sample_rate, chunk_size):
    spectral_centroids = []
    frequencies = np.linspace(0, sample_rate // 2, spectrogram.shape[0])

    for i in range(0, spectrogram.shape[1], chunk_size):
        chunk = spectrogram[:, i:i + chunk_size]
        if chunk.shape[1] == 0:
            continue
        # Normalize magnitude
        magnitudes = np.sum(chunk, axis=0)
        magnitudes[magnitudes == 0] = 1e-10  # Tránh chia cho 0

        # Compute centroid
        centroid = np.sum(frequencies[:, None] * chunk, axis=0) / magnitudes
        spectral_centroids.append(np.mean(centroid))  # Lấy trung bình cho mỗi đoạn

    return np.array(spectral_centroids)


# Hàm đọc file audio và spectrogram
def audio_read(file_path):
    # Đây là nơi bạn xử lý việc đọc file, ví dụ sử dụng librosa
    import librosa
    audio, sr = librosa.load(file_path, sr=22050)
    spectrogram = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    duration = len(audio) / sr
    return audio, spectrogram, duration


def evaluate_with_spectral_centroid(file_path, length=15, chunk_duration=3):
    results = []
    with open(file_path, mode='r', encoding='latin-1') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua dòng tiêu đề
        for row in reader:
            f = "dataset\\" + row[1] + ".mp3"

            # Kiểm tra nếu file không tồn tại
            if not os.path.exists(f):
                print(f"File not found: {f}")
                continue

            # Đọc file audio
            try:
                audio, spectrogram, duration = audio_read(f)
            except Exception as e:
                print(f"Error processing {row[1]}: {e}")
                continue

            # Các bước tính toán
            sample_rate = 22050
            chunk_size = chunk_duration * sample_rate // 512
            spectral_centroids = calculate_spectral_centroid(spectrogram, sample_rate, chunk_size)
            sorted_indices = np.argsort(spectral_centroids)[::-1]

            predictions = []
            for index in sorted_indices:
                if len(predictions) == 3:
                    break
                start_time = index * chunk_duration
                end_time = start_time + length
                is_overlap = any(
                    start_time < h_end and end_time > h_start for (h_start, h_end) in predictions
                )
                if not is_overlap:
                    predictions.append((start_time, end_time))

            # Nhãn đúng
            true_labels = []
            try:
                true_labels.append((convert_to_seconds(row[4]), convert_to_seconds(row[4]) + length))
                true_labels.append((convert_to_seconds(row[5]), convert_to_seconds(row[5]) + length))
                true_labels.append((convert_to_seconds(row[6]), convert_to_seconds(row[6]) + length))
            except Exception as e:
                print(f"Error reading labels for {row[1]}: {e}")
                continue

            # Tính metric
            precision, recall, f1_score = calculate_metrics(predictions, true_labels)
            results.append((precision, recall, f1_score))
            print(f"Processed {len(results)} songs.")

    # Tính trung bình Precision, Recall, F1
    PRE = sum([x[0] for x in results]) / len(results) if results else 0
    REC = sum([x[1] for x in results]) / len(results) if results else 0
    F1 = sum([x[2] for x in results]) / len(results) if results else 0
    print(f"Precision: {PRE:.2f}")
    print(f"Recall: {REC:.2f}")
    print(f"F1 Score: {F1:.2f}")


if __name__ == '__main__':
    file_path = 'dataset\\labelEdit.csv'
    evaluate_with_spectral_centroid(file_path=file_path, length=15)
