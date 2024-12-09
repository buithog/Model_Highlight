import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from model import MusicHighlighter
tf.compat.v1.disable_eager_execution()


# chkp.print_tensors_in_checkpoint_file('model/model', tensor_name='', all_tensors=True)

with tf.compat.v1.Session() as sess:
    model = MusicHighlighter()
    sess.run(tf.compat.v1.global_variables_initializer())
    model.saver.restore(sess, 'model/model')

    tf.compat.v1.saved_model.simple_save(sess, 'model_2',
                                         inputs={'x': model.x,
                                                 'pos_enc': model.pos_enc,
                                                 'num_chunk': model.num_chunk},
                                         outputs={'attn_score': model.attn_score})