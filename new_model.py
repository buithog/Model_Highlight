import tensorflow as tf
from tf_slim import layers as slim

class MusicHighlighter(object):
    def __init__(self):
        self.dim_feature = 64
        self.is_training = tf.compat.v1.placeholder_with_default(True, shape=(), name='is_training')
        self.bn_params = {'is_training': self.is_training, 'center': False, 'scale': True, 'updates_collections': None}

        # Placeholders for input data
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, None, 128], name='x')
        self.pos_enc = tf.compat.v1.placeholder(tf.float32, shape=[None, None, self.dim_feature * 4], name='pos_enc')
        self.num_chunk = tf.compat.v1.placeholder(tf.int32, name='num_chunk')
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 9], name='y')
        # Build the model
        self.build_model()

    def conv(self, inputs, filters, kernel, stride):
        dim = inputs.get_shape().as_list()[-2]
        return slim.conv2d(inputs, filters, [kernel, dim], [stride, dim],
                           activation_fn=tf.nn.relu,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=self.bn_params)

    def fc(self, inputs, num_units, dropout=0, act=tf.nn.relu):
        net = slim.fully_connected(inputs, num_units,
                                   activation_fn=act,
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=self.bn_params)

        net = tf.nn.dropout(net, rate=dropout)
        return net

    def attention(self, inputs, dim):
        outputs = self.fc(inputs, dim, 0, act=tf.nn.tanh)
        outputs = self.fc(outputs, 1, 0, act=None)
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
        attn_net = self.fc(attn_net, self.dim_feature * 4, 0)
        attn_net = self.fc(attn_net, self.dim_feature * 4, 0)
        self.attn_score = self.attention(attn_net, self.dim_feature * 4)

        net = self.fc(net, 512, dropout=0)
        net = self.fc(net, 256, dropout=0)
        chunk_predictions = self.fc(net, 9, dropout=0, act=tf.nn.sigmoid)
        self.overall_predictions = (tf.squeeze(tf.matmul(self.attn_score, chunk_predictions, transpose_a=True), axis=1))
        # self.overall_predictions = tf.nn.softmax(self.overall_predictions)
        epsilon = 1e-8
        self.loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.y, self.overall_predictions))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.overall_predictions))
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.math.log(self.overall_predictions), axis=1))
        # self.loss = tf.reduce_mean(tf.reduce_sum(-self.y * tf.math.log(overall_predictions) - (1 - self.y) * tf.math.log(1 - overall_predictions), axis=1))
        # Tính độ chính xác
        self.accuracy = self.calculate_accuracy(self.overall_predictions, self.y)
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    def calculate(self, sess, x, pos_enc, num_chunk):
        feed_dict = {self.x: x, self.pos_enc: pos_enc, self.num_chunk: num_chunk, self.is_training: False}
        return sess.run(self.attn_score, feed_dict=feed_dict)

    # def calculate_accuracy(self, predictions, labels):

    #     predict = tf.nn.softmax(predictions)
    #     predicted_labels = tf.argmax(predict, axis=1)

    #     correct_predictions = tf.equal(predicted_labels, tf.argmax(labels, axis=1))

    #     accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    #     return accuracy

    def calculate_accuracy(self, predictions, labels):
        # Chuyển predictions thành một one-hot vector
        max_indices = tf.argmax(predictions, axis=1)  # Lấy chỉ số của giá trị lớn nhất
        predicted_one_hot = tf.one_hot(max_indices, depth=labels.shape[1])  # Chuyển thành one-hot vector

        # So sánh với nhãn gốc
        correct_predictions = tf.reduce_all(tf.equal(predicted_one_hot, labels), axis=1)

        # Tính accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        return accuracy

    # def calculate_accuracy(self, predictions, labels):

    #     predicted_labels = tf.cast(predictions > 0.5, tf.float32)

    #     correct_predictions = tf.equal(predicted_labels, labels)

    #     accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    #     return accuracy
