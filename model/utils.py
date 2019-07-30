import skimage
import skimage.io
import skimage.transform
import numpy as np
import os
import tensorflow as tf
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]
# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]



# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))
#-------------------------------------------------------------------
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (256, 256))
    return resized_img

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def imageConvert(rgb):
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

def preprocess_image(img_path):
    with tf.Session() as sess:
        input_ = tf.placeholder(tf.float32, [None, 256, 256, 3])
        img = load_image(img_path)
        img = img.reshape((1, 256, 256, 3))
        #print(img.shape)
        feed_dict = {input_: img}
        bgr=imageConvert(input_)
        image=np.array(sess.run(bgr,feed_dict=feed_dict))
    return image

def saveImage(image):
    image_path="/home/liufei/Documents/20190310/cnn/params/image.txt"
    f=open(image_path,'w')
    list = []
    for k in range(3):
        for i in range(256):
            for j in range(256):
                list.append(image[0][i][j][k])
            np.savetxt(f, [list], fmt="%f", newline="\n")
            list = []
        f.write("\n")
    f.close()

def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
