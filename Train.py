import tensorflow as tf
from tf_slim import layers as slim
import numpy as np
import pandas as pd
from lib import audio_read, chunk, positional_encoding
import os

tf.compat.v1.disable_eager_execution()
class MusicHighlighter(object):
    def __init__(self):
        self.dim_feature = 64
        self.is_training = tf.compat.v1.placeholder_with_default(False, shape=(), name='is_training')
        self.bn_params = {'is_training': self.is_training, 'center': True, 'scale': True, 'updates_collections': None}

        # Placeholders for input data
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 128], name='x')
        self.pos_enc = tf.compat.v1.placeholder(tf.float32, shape=[None, None, self.dim_feature * 4], name='pos_enc')
        self.num_chunk = tf.compat.v1.placeholder(tf.int32, name='num_chunk')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 18], name='y')  # Multi-label placeholder

        # Build the model
        self.build_model()

    def conv(self, inputs, filters, kernel, stride):
        dim = inputs.get_shape().as_list()[-2]
        return slim.conv2d(inputs, filters, [kernel, dim], [stride, dim],
                           activation_fn=tf.nn.relu,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=self.bn_params)

    def fc(self, inputs, num_units, act=tf.nn.relu):
        return slim.fully_connected(inputs, num_units,
                                    activation_fn=act,
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=self.bn_params)

    def attention(self, inputs, dim):
        outputs = self.fc(inputs, dim, act=tf.nn.tanh)
        outputs = self.fc(outputs, 1, act=None)
        attn_score = tf.nn.softmax(outputs, axis=1)
        return attn_score

    def build_model(self):
        # 2D Conv. feature extraction
        net = tf.expand_dims(self.x, axis=3)
        net = self.conv(net, self.dim_feature, 3, 2)
        net = self.conv(net, self.dim_feature * 2, 4, 2)
        net = self.conv(net, self.dim_feature * 4, 4, 2)

        # Global max-pooling
        net = tf.squeeze(tf.reduce_max(net, axis=1), axis=1)

        # Restore shape [batch_size, num_chunk, dim_feature]
        net = tf.reshape(net, [1, self.num_chunk, self.dim_feature * 4])

        # Attention mechanism
        attn_net = net + self.pos_enc
        attn_net = self.fc(attn_net, self.dim_feature * 4)
        attn_net = self.fc(attn_net, self.dim_feature * 4)
        self.attn_score = self.attention(attn_net, self.dim_feature * 4)

        # Multi-label output prediction
        net = self.fc(net, 1024)
        chunk_predictions = self.fc(net, 18, act=tf.nn.sigmoid)
        overall_predictions = tf.squeeze(tf.matmul(self.attn_score, chunk_predictions, transpose_a=True), axis=1)
        epsilon = 1e-7
        sigmoid_preds = tf.sigmoid(overall_predictions)
        self.loss = -tf.reduce_mean(
            self.y * tf.math.log(sigmoid_preds + epsilon) + (1 - self.y) * tf.math.log(1 - sigmoid_preds + epsilon))
        # Tính độ chính xác
        self.accuracy = self.calculate_accuracy(overall_predictions, self.y)
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    def calculate(self, sess, x, pos_enc, num_chunk):
        feed_dict = {self.x: x, self.pos_enc: pos_enc, self.num_chunk: num_chunk, self.is_training: False}
        return sess.run(self.attn_score, feed_dict=feed_dict)


    def calculate_accuracy(self,predictions, labels):
        # Chuyển đổi đầu ra thành nhãn nhị phân
        predicted_labels = tf.cast(predictions > 0.5, tf.float32)
        correct_predictions = tf.reduce_sum(predicted_labels * labels)
        total_predictions = tf.reduce_sum(predicted_labels)

        # Tính độ chính xác (phòng trường hợp total_predictions bằng 0)
        accuracy = correct_predictions / tf.maximum(total_predictions, 1)
        return accuracy




# Training parameters
num_epochs = 10
batch_size = 1
learning_rate = 0.01

# Load song paths and labels
with open("song_paths.txt", "r") as f:
    song_paths = [line.strip() for line in f.readlines()]
print(song_paths[0])
labels = pd.read_csv("filtered_annotations.csv",skiprows=0).values
print(labels[0])
# Model setup
model = MusicHighlighter()
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(model.loss)

# Training loop
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_accuracy = 0
        num_samples = 0

        for i, file_name in enumerate(song_paths):
            if not os.path.exists(file_name):
                continue
            try:
                audio, spectrogram, duration = audio_read(file_name)
                n_chunk, remainder = np.divmod(duration, 3)
                chunk_spec = chunk(spectrogram, n_chunk)
                pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature * 4)
                y = np.expand_dims(labels[i], axis=0)

                feed_dict = {
                    model.x: chunk_spec,
                    model.pos_enc: pos,
                    model.num_chunk: n_chunk,
                    model.y: y,
                    model.is_training: True
                }

                _, loss_val, accuracy = sess.run([optimizer, model.loss, model.accuracy], feed_dict=feed_dict)

                epoch_loss += loss_val
                total_accuracy += accuracy
                num_samples += 1

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        # Tính độ chính xác và loss trung bình cho epoch
        average_accuracy = total_accuracy / num_samples if num_samples > 0 else 0
        average_loss = epoch_loss / num_samples if num_samples > 0 else 0

        print(f"Epoch: {epoch + 1}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}")

    # Save the model
    model.saver.save(sess, "model_emotify_50_dropout/model.ckpt")
