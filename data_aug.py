def central_scale_images(X_imgs, scales):
    IMAGE_SIZE = 144
    X_scale_data = []

    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([IMAGE_SIZE, IMAGE_SIZE], dtype = np.int32)

    for img_data in X_imgs:
        batch_img = np.expand_dims(img_data, axis = 0)
        tf_img = tf.image.crop_and_resize(batch_img, boxes, box_ind, crop_size)
        scaled_imgs = sess.run(tf_img)
        X_scale_data.extend(scaled_imgs)

    X_scale_data.extend(X_imgs)
    X_scale_data = np.array(X_scale_data, dtype = np.float32)

    return X_scale_data

def rotate_images(X_imgs, start_angle, end_angle, n_images):
    from math import pi

    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    for index in range(n_images):
        degrees_angle = start_angle + index * iterate_at
        radian_value = degrees_angle * pi / 180  # Convert to radian
        radian_arr = [radian_value] * len(X_imgs)
        tf_img = tf.contrib.image.rotate(X_imgs, radian_arr)
        rotated_imgs = sess.run(tf_img)
        X_rotate.extend(rotated_imgs)

    X_rotate.extend(X_imgs)
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

def flip_images(X_imgs):
    X_flip = []

    for img in X_imgs:
        tf_img1 = tf.image.flip_left_right(img)
        tf_img2 = tf.image.flip_up_down(img)
        tf_img3 = tf.image.transpose_image(img)
        flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3])
        X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip

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

def add_gaussian_noise(X_imgs):
    import cv2
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
    
