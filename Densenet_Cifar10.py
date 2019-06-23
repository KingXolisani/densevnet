import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from PIL import Image
import matplotlib.image as mpimg

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

IMAGE_SIZE = 224

def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE), 
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)

    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data

def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data

# Produce each image at scaling of 90%, 75% and 60% of original image.
scaled_imgs = central_scale_images(X_imgs, [0.90, 0.75, 0.60])

from math import ceil, floor

def get_translate_parameters(index):
    if index == 0: # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1: # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype = np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype = np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2: # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE))
    else: # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype = np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype = np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype = np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3),
				    dtype = np.float32)
            X_translated.fill(1.0) # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
			 w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype = np.float32)
    return X_translated_arr

translated_imgs = translate_images(X_imgs)

def rotate_images(X_imgs):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

rotated_imgs = rotate_images(X_imgs)

from math import pi

def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (None, IMAGE_SIZE, IMAGE_SIZE, 3))
    radian = tf.placeholder(tf.float32, shape = (len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict = {X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
rotated_imgs = rotate_images(X_imgs, -90, 90, 14)

def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

flipped_images = flip_images(X_imgs)

def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy

salt_pepper_noise_imgs = add_salt_pepper_noise(X_imgs)

import cv2

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs

gaussian_noise_imgs = add_gaussian_noise(X_imgs)

def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype = np.int32)
    return vertices

def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix

def perspective_transform(X_img):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
    return warped_img

perspective_img = perspective_transform(X_img)

def read_dataset(hf5):
    import numpy as np
    import h5py

    hf = h5py.File(hf5,'r')
    x_train = hf.get('x_train')
    y_train = hf.get('y_train')

    return x_train, y_train

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

    def dense_block(self, input_x, filters, layer_name):
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
x_train, y_train  = read_dataset(dataset)
x_train = x_train/255 # Normalise data

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
                batch_y = y_train[pre_index:pre_index+ batch_size]
            else:
                batch_x = x_train[pre_index:]
                batch_y = y_train[pre_index:]

            # preprocess: ensure each entry in label batch is in [0, num_classes)
            for batch in range(len(batch_y)):
                for y in range(len(batch_y[batch])):
                    for x in range(len(batch_y[batch][y])):
                        if batch_y[batch][y][x][0] > 20.0:
                            batch_y[batch][y][x][0] = 0.0

            train_feed_dict = {
                image_ph: batch_x,
                mask_ph: batch_y,
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
