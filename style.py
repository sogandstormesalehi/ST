import functools
import os
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time


def crop_center(image):
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) 
    offset_x = max(shape[2] - shape[1], 0)
    image = tf.image.crop_to_bounding_box(
    image, offset_y, offset_x, new_shape, new_shape)
    return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    image_to_return = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    image_to_return = crop_center(image_to_return)
    image_to_return = tf.image.resize(image_to_return, image_size, preserve_aspect_ratio=True)
    return image_to_return

def show_n(images, titles=('',)):
    length = len(images)
    image_sizes = [image.shape[1] for image in images]
    weight = (image_sizes[0] * 6)
    plt.figure(figsize=(weight * length, w))
    gridspecc = gridspec.GridSpec(1, length, width_ratios=image_sizes)
    for i in range(length):
        plt.subplot(gridspecc[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()


content_image_url = 'https://i.pinimg.com/564x/5f/7a/ae/5f7aae33c114ef44410febb08e683b7a.jpg' 
style_image_url = 'https://i.pinimg.com/564x/96/9b/39/969b3960fb339259530026a44974a676.jpg'
output_image_size = 384 
content_img_size = (output_image_size, output_image_size)
style_img_size = (384, 384) 
content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')




hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
outputs = hub_module(content_image, style_image)
stylized_image = outputs[0]
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]
show_n([content_image, style_image, stylized_image], titles=['content', 'style', 'transfered'])
