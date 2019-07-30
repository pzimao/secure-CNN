from utils import load_image
import tensorflow as tf
import numpy as np
VGG_MEAN = [103.939, 116.779, 123.68]
def preprocess_image(rgb):
    rgb_scaled = rgb * 255.0

    # Convert RGB to BGR
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [256, 256, 1]
    assert green.get_shape().as_list()[1:] == [256, 256, 1]
    assert blue.get_shape().as_list()[1:] == [256, 256, 1]
    bgr = tf.concat(axis=3, values=[
        (blue - VGG_MEAN[0]) / 255,
        (green - VGG_MEAN[1]) / 255,
        (red - VGG_MEAN[2]) / 255,
    ])
    assert bgr.get_shape().as_list()[1:] == [256, 256, 3]
    return bgr

def ImageSave(img_path):
    with tf.Session() as sess:
        input_ = tf.placeholder(tf.float32, [None, 256, 256, 3])
        img_path = "/home/liufei/Documents/20190310/cnn/Data/corel_data/" + str(300) + ".jpg"
        img = load_image(img_path)
        img = img.reshape((1, 256, 256, 3))
        print(img.shape)
        feed_dict = {input_: img}
        bgr=preprocess_image(input_)
        image=np.array(sess.run(bgr,feed_dict=feed_dict))
    return image

import os
CUR_PATH = r'/home/liufei/Documents/20190310/cnn/test'
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

del_file(CUR_PATH)
def main():
    #ImageSave()
    del_file(CUR_PATH)

if __name__ == '__main__':
    main()
