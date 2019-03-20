#!/usr/bin/env python3

import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
#from scipy.misc import imread
from imageio import imread
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
########
# PATHS
########
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("path", type=str, 
        help='Path to the generated images or to .npz statistic files')
parser.add_argument("--gpu", default="", type=str,
        help='GPU to use (leave blank for CPU only)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


data_path = args.path # set path to training set images
output_path = './data_sets/'+data_path.split('/')[-1]+'.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.png'))


#images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(image_list))

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    act = fid.get_activations_from_files(image_list, sess, batch_size=100)
    np.savez_compressed(output_path, act=act)
print("finished")
