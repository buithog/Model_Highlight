import tensorflow as tf

tf.compat.v1.enable_control_flow_v2()
# Chuyển đổi SavedModel sang định dạng TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('model_2')
tflite_model = converter.convert()
# Lưu mô hình TensorFlow Lite
with open('music_highlighter.tflite', 'wb') as f:
    f.write(tflite_model)