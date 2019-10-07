import tensorflow as tf
from numpy import *
from random import randint

max_length = 3
batch_size = 5

targets = array([[1 for _ in range(max_length)] for _ in range(batch_size)])
logits = array([[[randint(0,10)/10,randint(0,10)/10] for _ in range(max_length)] for _ in range(batch_size)])
sequence_length = array([randint(1,max_length) for _ in range(batch_size)])
print(targets)
print(logits)
print(shape(targets))
print(shape(logits))
print('-'*20)
targets = tf.convert_to_tensor(targets, dtype = tf.int32)
logits = tf.convert_to_tensor(logits, dtype = tf.float32)
print(targets)
print(logits)
print('='*20)
loss_before_mask = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
loss_mask = tf.sequence_mask(tf.to_int32(sequence_length), tf.to_int32(max_length))
loss_after_mask = loss_before_mask * tf.to_float(loss_mask)
	
with tf.Session() as sess:
	print(loss_mask)
	print("sequence length: "+str(sequence_length))
	print("sequence mask:")
	print(sess.run(loss_mask))
	print('+'*20)
	print("loss before mask:")
	print(loss_before_mask)
	print(sess.run(loss_before_mask))
	print('*'*20)
	print("loss after mask:")
	# Mask out the losses we don't care about
	print(loss_after_mask)
	print(sess.run(loss_after_mask))