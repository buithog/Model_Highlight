from new_model import MusicHighlighter
from lib import *
import tensorflow as tf
import numpy as np
import os
import soundfile as sf
import numpy as np
import librosa
os.environ["CUDA_VISIBLE_DEVICES"] = ''

def extract(fs, length=30, save_score=True, save_thumbnail=True, save_wav=True):
    with tf.compat.v1.Session() as sess:
        model = MusicHighlighter()
        sess.run(tf.compat.v1.global_variables_initializer())
        model.saver.restore(sess, 'model_emotify_50_dropout/model')
        for f in fs:
            name = os.path.split(f)[-1][:-4]
            audio, spectrogram, duration = audio_read(f)
            n_chunk, remainder = np.divmod(duration, 3)
            chunk_spec = chunk(spectrogram, n_chunk)
            pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature*4)

            n_chunk = n_chunk.astype('int')
            chunk_spec = chunk_spec.astype('float')
            pos = pos.astype('float')

            attn_score = model.calculate(sess=sess, x=chunk_spec, pos_enc=pos, num_chunk=n_chunk)
            attn_score = np.repeat(attn_score, 3)
            attn_score = np.append(attn_score, np.zeros(remainder))

            # score
            attn_score = attn_score / attn_score.max()
            # if save_score:
            #     np.save('{}_score.npy'.format(name), attn_score)

            # thumbnail
            attn_score = attn_score.cumsum()
            attn_score = np.append(attn_score[length], attn_score[length:] - attn_score[:-length])
            index = np.argmax(attn_score)
            # highlight = [index, index+length]
            # print(highlight)
            sorted_indices = np.argsort(attn_score)[::-1]
            predictions = []
            used_indices = set()
            for index in sorted_indices:
                if (len(predictions) == 3):
                    break
                is_overlap = any(index < h_end and index + length > h_start for (h_start, h_end) in predictions)
                if not is_overlap:
                    predictions.append([index, index + length])
            print(predictions)
            # if save_thumbnail:
            #     np.save('{}_highlight.npy'.format(name), highlight)
            #output
            # if save_wav:
            #     sf.write('{}_audio.mp3'.format(name), audio[highlight[0]*22050:highlight[1]*22050], 22050)

if __name__ == '__main__':
    fs = ['Aylex - Summer Sound (freetouse.com).mp3'] # list
    extract(fs, length=15, save_score=True, save_thumbnail=True, save_wav=True)
