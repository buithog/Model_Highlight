import numpy as np
import tensorflow as tf
import soundfile as sf
from lib import *
from model import MusicHighlighter
import librosa
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ''


def extract_tflite(fs, length=15, save_score=True, save_thumbnail=True, save_wav=True):
    # Tải mô hình TFLite
    interpreter = tf.lite.Interpreter(model_path="music_highlighter.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for f in fs:
        name = os.path.split(f)[-1][:-4]
        audio, spectrogram, duration = audio_read(f)
        n_chunk, remainder = np.divmod(duration, 3)
        chunk_spec = chunk(spectrogram, n_chunk)
        pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=256)


        # Chuyển đổi dữ liệu đầu vào sang kiểu float32
        n_chunk = np.array(n_chunk, dtype=np.int32)
        chunk_spec = chunk_spec.astype(np.float32)
        pos = pos.astype(np.float32)

        # Resize tensor input if needed
        interpreter.resize_tensor_input(input_details[0]['index'], pos.shape)
        interpreter.resize_tensor_input(input_details[2]['index'], chunk_spec.shape)
        interpreter.allocate_tensors()

        # Đặt giá trị đầu vào cho mô hình TFLite
        interpreter.set_tensor(input_details[0]['index'], pos)  # Đặt giá trị cho pos_enc
        interpreter.set_tensor(input_details[1]['index'], n_chunk)  # Đặt giá trị cho num_chunk
        interpreter.set_tensor(input_details[2]['index'], chunk_spec)  # Đặt giá trị cho x

        # Chạy mô hình TFLite
        interpreter.invoke()

        # Lấy kết quả đầu ra từ mô hình TFLite
        attn_score = interpreter.get_tensor(output_details[0]['index'])

        # Xử lý kết quả
        attn_score = np.repeat(attn_score, 3)
        attn_score = np.append(attn_score, np.zeros(remainder))

        # score
        attn_score = attn_score / attn_score.max()
        # if save_score:
        #     np.save('{}_score_lite.npy'.format(name), attn_score)

        # thumbnail
        attn_score = attn_score.cumsum()
        attn_score = np.append(attn_score[length], attn_score[length:] - attn_score[:-length])
        # index = np.argmax(attn_score)
        # highlight = [index, index + length]
        # print(highlight)
        sorted_indices = np.argsort(attn_score)[::-1]
        predictions = []
        used_indices = set()

        for index in sorted_indices:
            if len(predictions) == 3:
                break

            is_overlap = any(index < h_end and index + length > h_start for (h_start, h_end) in predictions)

            if not is_overlap:
                predictions.append([index, index + length])

        highlight = [h_start for h_start, h_end in predictions]
        print(highlight)
        # for i in range(len(highlight)):
        #     print(attn_score[highlight[i]])
        #     print('\n')
        # # Vẽ biểu đồ và làm nổi bật phần highlight
        # plt.figure(figsize=(10, 6))
        # plt.plot(attn_score, label='Attention Score')
        # # Tô màu phần highlight
        # for i in range(len(highlight)):
        #     plt.axvspan(highlight[i], highlight[i] + 15, color='orange', alpha=0.5, label='Highlight')
        # # Các chi tiết bổ sung cho biểu đồ
        # plt.title('Attention Score with Highlight')
        # plt.xlabel('Time')
        # plt.ylabel('Score')
        # plt.legend()
        # plt.show()
        # if save_thumbnail:
        #     np.save('{}_highlight_lite.npy'.format(name), highlight)
        #
        # # output audio
        # if save_wav:
        #     sf.write('{}_audio_lite.mp3'.format(name), audio[highlight[0] * 22050:highlight[1] * 22050], 22050)


if __name__ == '__main__':
    fs = ['Illenium.mp3']  # list
    extract_tflite(fs, length=30, save_score=True, save_thumbnail=True, save_wav=True)
