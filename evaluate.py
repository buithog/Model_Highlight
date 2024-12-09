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


def calculate_metrics(predictions: List[Tuple[int, int]], true_labels: List[Tuple[int, int]]) -> Tuple[float, float, float]:
    max_overlap = 0
    for pred in predictions:
        for true_label in true_labels:
            max_overlap = max(max_overlap, overlap(pred, true_label))

    precision = max_overlap / (predictions[0][1] - predictions[0][0]) if predictions else 0
    recall = max_overlap / (true_labels[0][1] - true_labels[0][0]) if true_labels else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def evaluate(file_path, length=15):
    results = []
    with tf.compat.v1.Session() as sess:
        model = MusicHighlighter()
        sess.run(tf.compat.v1.global_variables_initializer())
        model.saver.restore(sess, 'model/model')

        with open(file_path, mode='r', encoding='latin-1') as file:
            reader = csv.reader(file)
            next(reader)  # Bỏ qua tiêu đề CSV

            for row in reader:
                f = f"dataset\\{row[1]}.mp3"
                audio, spectrogram, duration = audio_read(f)
                n_chunk, remainder = np.divmod(duration, 3)
                chunk_spec = chunk(spectrogram, n_chunk)
                pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature * 4)

                n_chunk = n_chunk.astype('int')
                chunk_spec = chunk_spec.astype('float')
                pos = pos.astype('float')

                attn_score = model.calculate(sess=sess, x=chunk_spec, pos_enc=pos, num_chunk=n_chunk)
                attn_score = np.repeat(attn_score, 3)
                attn_score = np.append(attn_score, np.zeros(remainder))
                attn_score = attn_score / attn_score.max()

                attn_score = attn_score.cumsum()
                attn_score = np.append(attn_score[length], attn_score[length:] - attn_score[:-length])

                sorted_indices = np.argsort(attn_score)[::-1]
                predictions = []
                for index in sorted_indices:
                    if len(predictions) == 3:
                        break
                    start = index
                    end = start + length
                    if all(not (p[0] < end and p[1] > start) for p in predictions):
                        predictions.append((start, end))

                true_labels = []
                for i in range(4, 7):
                    if row[i]:
                        start = convert_to_seconds(row[i])
                        true_labels.append((start, start + length))

                precision, recall, f1_score = calculate_metrics(predictions, true_labels)
                results.append((precision, recall, f1_score))

    # Tính tổng kết các chỉ số
    PRE = sum([x[0] for x in results]) / len(results) if results else 0
    REC = sum([x[1] for x in results]) / len(results) if results else 0
    F1 = sum([x[2] for x in results]) / len(results) if results else 0
    print(f"Precision: {PRE:.2f}")
    print(f"Recall: {REC:.2f}")
    print(f"F1 Score: {F1:.2f}")


    # # In chi tiết từng bài hát
    for i, (precision, recall, f1_score) in enumerate(results):
        print(f"Music {i + 1}: Precision = {precision}, Recall = {recall}, F1-score = {f1_score}")

if __name__ == '__main__':
    file_path = 'dataset\\labelEdit.csv'
    evaluate(file_path=file_path, length=15)
