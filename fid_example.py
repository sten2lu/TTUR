#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid

import tensorflow as tf

#from scipy.misc import imread
from imageio import imread
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Paths
image_path = './data_sets/single_images/' # set path to some generated images
stats_path = './data_sets/fid_stats_imagenet_valid.npz' # training set statistics
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(None) # download inception network
print("ok")


# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(image_path, '*.png'))
#print(image_list)
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(images))

# load precalculated training set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)
print("ok")

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
