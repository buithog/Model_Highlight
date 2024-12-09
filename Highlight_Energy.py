from typing import List, Tuple
import csv
from model import MusicHighlighter
from lib import *
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ''


def convert_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + int(seconds)
    return total_seconds


def overlap(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> int:
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start)


def calculate_metrics(predictions: List[Tuple[int, int]], true_labels: List[Tuple[int, int]]) -> Tuple[
    float, float, float]:
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


def calculate_energy(spectrogram, chunk_size):
    energy = []
    for i in range(0, spectrogram.shape[1], chunk_size):
        chunk = spectrogram[:, i:i + chunk_size]
        chunk_energy = np.sum(chunk ** 2)  # Tổng bình phương biên độ
        energy.append(chunk_energy)
    return np.array(energy)


def evaluate_with_energy(file_path, length=15, chunk_duration=3):
    results = []
    with open(file_path, mode='r', encoding='latin-1') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua dòng tiêu đề
        for row in reader:
            # Kiểm tra file audio
            f = "dataset\\" + row[1] + ".mp3"
            if not os.path.exists(f):
                print(f"File not found: {f}")
                continue

            # Đọc file audio
            try:
                audio, spectrogram, duration = audio_read(f)
            except Exception as e:
                print(f"Error processing {row[1]}: {e}")
                continue

            # Tính năng lượng
            sample_rate = 22050
            chunk_size = chunk_duration * sample_rate // 512
            energy = calculate_energy(spectrogram, chunk_size)
            sorted_indices = np.argsort(energy)[::-1]

            # Chọn 3 khoảng năng lượng cao nhất
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

            # Đọc nhãn đúng từ file CSV
            try:
                true_labels = [
                    (convert_to_seconds(row[4]), convert_to_seconds(row[4]) + length),
                    (convert_to_seconds(row[5]), convert_to_seconds(row[5]) + length),
                    (convert_to_seconds(row[6]), convert_to_seconds(row[6]) + length),
                ]
            except Exception as e:
                print(f"Error reading labels for {row[1]}: {e}")
                continue

            # Tính Precision, Recall, F1
            precision, recall, f1_score = calculate_metrics(predictions, true_labels)
            results.append((precision, recall, f1_score))
            print(f"Processed {len(results)} songs.")

    # Tính trung bình Precision, Recall, F1
    if results:
        PRE = sum([x[0] for x in results]) / len(results)
        REC = sum([x[1] for x in results]) / len(results)
        F1 = sum([x[2] for x in results]) / len(results)
        print(f"Precision: {PRE:.2f}")
        print(f"Recall: {REC:.2f}")
        print(f"F1 Score: {F1:.2f}")
        # for i, (precision, recall, f1_score) in enumerate(results):
        #     print(f"Music {i + 1}: Precision = {precision}, Recall = {recall}, F1-score = {f1_score}")
    else:
        print("No valid songs processed.")



if __name__ == '__main__':
    file_path = 'dataset\\labelEdit.csv'
    evaluate_with_energy(file_path=file_path, length=15)
