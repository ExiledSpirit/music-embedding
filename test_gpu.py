import os
import tensorflow as tf

# Show all TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

print("TensorFlow version:", tf.__version__)

# Print available GPUs
gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)

# Force placement logging
tf.debugging.set_log_device_placement(True)

# Dummy operation
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a + b
print("Result:", c)