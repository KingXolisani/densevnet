import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from PIL import Image
import matplotlib.image as mpimg

import data_aug

# Hyperparameter
dataset = 'VOC_dataset.h5'
data_size = 12031
num_classes = 21
init_learning_rate = 0.001
epsilon = 1e-4 # AdamOptimizer epsilon
dropout_rate = 0.5
batch_size = 10
iteration = 1204 # batch_size * iteration = data_set_number
test_iteration = 10
total_epochs = 625

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

""" utilities/ helper fuctions """

def data_augmenation(imgs):
    data_out = []
    data_out.append(data_aug.scale(imgs, [0.6, 0.8, 0.9]))
    data_out.append(data_aug.rotate(imgs, -5, 5, 3))
    #data_out.append(data_aug.add_noise(imgs))

    return data_out

def read_dataset(hf5):
    import numpy as np
    import h5py

    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y_train72 = hf.get('y_train72')
    y_train144 = hf.get('y_train144')

    return x_train, y_train72, y_train144

""" Model layers fuctions """

def conv_layer(input_x, filters, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input_x, use_bias=False, filters=filters, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    return global_avg_pool(x, name='Global_avg_pooling')

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


""" Evaluation of model fuctions """

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / 10.0
        test_acc += acc_ / 10.0

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

""" Loss fuction definitions """

def xentropy_loss(logits, labels, num_classes):
    labels = tf.cast(labels, tf.int32)
    logits = tf.reshape(logits, [tf.shape(logits)[0], -1, num_classes])
    labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name="loss")

    return loss

def calculate_iou(mask, prediction, num_classes):
    mask = tf.reshape(tf.one_hot(tf.squeeze(mask), depth = num_classes), [tf.shape(mask)[0], -1, num_classes])
    prediction = tf.reshape(prediction, shape=[tf.shape(prediction)[0], -1, num_classes])
    iou, update_op = tf.metrics.mean_iou(tf.argmax(prediction, 2), tf.argmax(mask, 2), num_classes)

    return iou, update_op

""" Model class """

class DenseNet():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, filters, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filters=4*filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filters=filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, kernel, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            in_channel = x.shape[-1]
            x = conv_layer(x, filters=24, kernel=kernel, layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def upsample_layer(self, bottom, n_channels, name, upscale_factor):
        kernel_size = 2*upscale_factor - upscale_factor%2
        stride = upscale_factor
        strides = [1, stride, stride, 1]

        def get_bilinear_filter(filter_shape, upscale_factor):
            bilinear = np.zeros([filter_shape[0],filter_shape[1]])
            weights = np.zeros(filter_shape)

            for i in range(filter_shape[2]):
                weights[:, :, i, i] = bilinear

            init = tf.constant_initializer(value=weights,dtype=tf.float32)

            bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init, shape=weights.shape)
            return bilinear_weights

        with tf.variable_scope(name):
            # Shape of the bottom tensor
            in_shape = tf.shape(bottom)

            h = in_shape[1] * upscale_factor
            w = in_shape[2] * upscale_factor

            new_shape = [in_shape[0], h, w, n_channels]
            output_shape = tf.stack(new_shape)

            filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

            weights = get_bilinear_filter(filter_shape,upscale_factor)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv

    def dense_block(self, input_x, filters, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()

            x = self.bottleneck_layer(input_x, filters, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, filters, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    """ Models """
    def Dense_net(self, input_x):

       Conv_Down0 = self.transition_layer(input_x, [5,5], scope='Conv_Down0')

       Dense1 = self.dense_block(input_x=Conv_Down0, filters=4, nb_layers=5, layer_name='dense_1')
       Conv_Down1 = self.transition_layer(Dense1, [3,3], scope='Conv_Down1')

       Dense2 = self.dense_block(input_x=Conv_Down1, filters=8, nb_layers=10, layer_name='dense_2')
       Conv_Down2 = self.transition_layer(Dense2, [3,3], scope='Conv_Down2')

       Dense3 = self.dense_block(input_x=Conv_Down2, filters=16, nb_layers=10, layer_name='dense_3')

       Conv1 = conv_layer(Dense3, filters= 24, kernel=[3,3], stride=1,layer_name='Conv1')
       upx4 = self.upsample_layer(Conv1, Conv1.get_shape()[-1], 'upx4', 4)

       Conv2 = conv_layer(Dense2, filters= 24, kernel=[3,3], stride=1,layer_name='Conv2')
       upx2 = self.upsample_layer(Conv2, Conv2.get_shape()[-1], 'upx2', 2)

       Conv3 = conv_layer(Dense1, filters= 24, kernel=[3,3], stride=1,layer_name='Conv3')

       Merge1 = Concatenation([Conv3,upx2,upx4])

       Conv4 = conv_layer(Merge1, filters= num_classes , kernel=[3,3], stride=1,layer_name='Conv4')

       return Conv4


""" Building tensorflow graphs """
# Loading dataset
x_train, y_train72, y_train144  = read_dataset(dataset)
x_train = x_train # Normalise data

# Creating data placeholders
image_ph = tf.placeholder(tf.float32, shape=[None, 144, 144, 3])
mask_ph = tf.placeholder(tf.int32, shape=[None, 72, 72, 1])
training = tf.placeholder(tf.bool, shape=[])

# Init model and loss function
logits = DenseNet(x=image_ph, training=training).model
loss = tf.reduce_mean(xentropy_loss(logits, mask_ph, num_classes))

with tf.variable_scope("mean_iou_train"):
    iou, iou_update = calculate_iou(mask_ph, logits, num_classes)

# Init Optimizer function
optimizer = tf.train.AdamOptimizer(init_learning_rate, epsilon=epsilon)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = optimizer.minimize(loss)

running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mean_iou_train")

reset_iou = tf.variables_initializer(var_list=running_vars)

saver = tf.train.Saver(max_to_keep=20)


""" Running graphs in session  """
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.local_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, iteration + 1):
            if pre_index+batch_size < data_size:
                batch_x = x_train[pre_index:pre_index+ batch_size]
                batch_y = y_train72[pre_index:pre_index+ batch_size]
            else:
                batch_x = x_train[pre_index:]
                batch_y = y_train72[pre_index:]

            batch_x_aug = data_augmenation (batch_x)
            batch_y_aug = data_augmenation (batch_y)

            # preprocess: ensure each entry in label batch is in [0, num_classes)
            for batch in range(len(batch_y_aug)):
                for y in range(len(batch_y_aug[batch])):
                    for x in range(len(batch_y_aug[batch][y])):
                        if batch_y_aug[batch][y][x][0] > 20.0:
                            batch_y_aug[batch][y][x][0] = 0.0

            train_feed_dict = {
                image_ph: batch_x_aug,
                mask_ph: batch_y_aug,
                training : True
            }

            cost,_,_  = sess.run([loss, opt, iou_update], feed_dict=train_feed_dict)
            train_iou = sess.run(iou, feed_dict=train_feed_dict)

            train_loss += cost
            train_acc += train_iou
            pre_index += batch_size

            if step == iteration :

                train_loss /= iteration # average loss
                train_acc /= iteration # average accuracy

                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

                #test_acc, test_loss, test_summary = Evaluate(sess)

                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                #summary_writer.add_summary(summary=test_summary, global_step=epoch)
                summary_writer.flush()

                line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f \n" % (
                    epoch, total_epochs, train_loss, train_acc) #, test_loss, test_acc)
                print(line)

                with open('logs.txt', 'a') as f :
                    f.write(line)

        saver.save(sess=sess, save_path='./model/dense.ckpt')
