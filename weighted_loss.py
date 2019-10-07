import tensorflow as tf
from numpy import *
from random import randint

max_length = 2
batch_size = 3

label = array([[[0,0] for _ in range(max_length)] for _ in range(batch_size)])
for i in range(batch_size):
	for j in range(max_length):
		label[i][j][randint(0,1)]=1

targets = argmax(label, axis=2)
logits = array([[[randint(0,10)/10,randint(0,10)/10] for _ in range(max_length)] for _ in range(batch_size)])


print("label")
print(label)
print("targets")
print(targets)
print("logits")
print(logits)

label = tf.convert_to_tensor(label, dtype=tf.float32)
targets = tf.convert_to_tensor(targets, dtype=tf.int32)
logits = tf.convert_to_tensor(logits, dtype=tf.float32)
class_weight = tf.constant([1.0, 0.2], shape=[1,2], dtype=tf.float32)

loss_before_weighted = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels=targets)
weighted_label = tf.transpose( tf.matmul(tf.reshape(label,[-1,2]), tf.transpose(class_weight)) ) #shape [1,num_steps*batch_size]
weighted_label = tf.reshape(weighted_label,[batch_size,max_length])
loss_after_weighted = tf.multiply(weighted_label, loss_before_weighted)

with tf.Session() as sess:
	print("Class Weight:")
	print(sess.run(class_weight))
	print("\nweighted Label:")
	print(sess.run(weighted_label))
	print("\nLoss before weighted:")
	print(sess.run(loss_before_weighted))
	print("\nLoss after weighted:")
	print(sess.run(loss_after_weighted))
