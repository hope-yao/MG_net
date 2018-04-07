import tensorflow as tf


def new_weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def new_bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
