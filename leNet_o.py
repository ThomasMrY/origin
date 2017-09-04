import tensorflow as tf
import ops
import batch as bn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 128
Train_steps = 10000
def add_full_layer(input_x,layer_numble,node_num,activation_function = None):
	layer_name = 'full_connected_layer%s' %layer_numble
	with tf.name_scope(layer_name):
		with tf.name_scope('Weight'):
			Weight = tf.Variable(tf.random_uniform([input_x.shape.as_list()[1]]+[node_num],-0.1,0.1),dtype = tf.float32,name = "W")
			tf.summary.histogram(layer_name+'/Weight',Weight)
		with tf.name_scope('Bias'):
			bias = tf.Variable(tf.constant(0.1,shape = [node_num]),dtype = tf.float32,name = "b")
			tf.summary.histogram(layer_name+'/bias',bias)
		with tf.name_scope('W_x_add_b'):
			temp = tf.add(tf.matmul(input_x,Weight),bias)
		if(activation_function is None):
			output_y = temp
		else:
			output_y = activation_function(temp)
		tf.summary.histogram(layer_name+'/output',output_y)
		return output_y

def add_cov_layer(input_x,layer_numble,filter_size,activation_function = None):
	layer_name = "convolutional_layer%s" %layer_numble
	with tf.name_scope(layer_name):
		with tf.name_scope("Filter"):
			Filter = tf.Variable(tf.random_uniform(filter_size,-0.1,0.1),dtype = tf.float32,name = "filter")
			tf.summary.histogram(layer_name+'/filter',Filter)
		with tf.name_scope("Bias"):
			bias = tf.Variable(tf.constant(0.1,shape = [filter_size[3]]),dtype = tf.float32,name = "bias")
			tf.summary.histogram(layer_name+'/bias',bias)
		with tf.name_scope("convolution"):
			conv = tf.nn.conv2d(input_x,Filter,strides=[1, 1, 1, 1], padding='VALID')
		with tf.name_scope('conv_add_b'):
			temp = tf.add(conv,bias)
		if(activation_function is None):
			output_y = temp
		else:
			output_y = activation_function(temp)
		tf.summary.histogram(layer_name+'/output',output_y)
		return output_y

def add_max_pool(input_x,layer_numble,k_size):
	layer_name = 'pool_layer%s' %layer_numble
	with tf.name_scope(layer_name):
		output_y = tf.nn.max_pool(input_x,k_size,strides=[1, 2, 2, 1],padding = 'VALID')
		tf.summary.histogram(layer_name+'/output',output_y)
	return output_y

def stack2line(input_x,layer_numble):
	layer_name = 'stack2line%s' %layer_numble
	with tf.name_scope(layer_name):
		output_y = tf.reshape(input_x,[-1,tf.cast(input_x.shape[1]*input_x.shape[2]*input_x.shape[3],tf.int32)])
	return output_y
########################################################
#the entrence of the data
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32,[None,784],name = 'x_input')
	y_ = tf.placeholder(tf.float32,[None,10],name = 'y_input')
	x_imgs = tf.reshape(x,[-1,28,28,1])

keep_prob = tf.placeholder(tf.float32)
########################################################

########################################################
#the structure of the cnn:
layer1 = add_cov_layer(x_imgs,1,[5,5,1,16],activation_function = tf.nn.relu)
layer2 = add_max_pool(layer1,2,[1,2,2,1])
layer3 = add_cov_layer(layer2,3,[5,5,16,6],activation_function = tf.nn.relu)
layer4 = add_max_pool(layer3,4,[1,2,2,1])
layer5 = stack2line(layer4,5)
layer5_bn = bn.batch_norm(layer5)
# layer6 = add_full_layer(layer5,6,400,activation_function = ops.selu)
# layer7 = add_full_layer(layer6,7,80,activation_function = ops.selu)
layer6 = add_full_layer(layer5_bn,6,400,activation_function = tf.nn.relu)
layer7 = add_full_layer(layer6,7,80,activation_function = tf.nn.relu)
layer7_drop = tf.nn.dropout(layer7,keep_prob)
y = add_full_layer(layer7_drop,8,10,activation_function =tf.nn.softmax)
#########################################################

#########################################################
#define of loss_function
with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]),name='cross_entropy')
	tf.summary.scalar('cross_entropy',cross_entropy)
#########################################################

#########################################################
#define train optimizer
with tf.name_scope('train'):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#########################################################

#########################################################
#culculate the accuracy
with tf.name_scope('accuracy'):
	prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32),name = 'accuracy')
	tf.summary.scalar('accuracy',accuracy)
#########################################################

save=tf.train.Saver()

with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	merge = tf.summary.merge_all()
	writer = tf.summary.FileWriter('log/',sess.graph)
	for i in range(Train_steps):
		batch = mnist.train.next_batch(batch_size)
		sess.run(train_step,feed_dict = {x:batch[0],y_:batch[1],keep_prob:0.5})
		if(i%100 == 0):
			accuracy_batch = sess.run(accuracy,feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
			result = sess.run(merge,feed_dict={x:batch[0],y_:batch[1],keep_prob : 1.0})
			writer.add_summary(result,i)
			loss_batch = sess.run(cross_entropy,feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
			print('after %d train steps the loss on the batch data is %g and the accuracy on batch data is %g'%(i,loss_batch,accuracy_batch))
	save_path=save.save(sess,'net_data/le_net/cnn_theta.ckpt')
	print('save_path is ',save_path)