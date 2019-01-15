#!/usr/bin/python

import numpy as np
from PIL import Image
import caffe

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
caffe.set_mode_gpu()

# usage: python test_caffe.py caffe  "bag2019-01-02-11-33-05_100s_front" true

OUTPUT_NAME = "out_label"

check_point  = sys.argv[1]
dataset_name = "image_2"
crop = "false" # if to crop the half image
if len(sys.argv) >= 3:
    dataset_name = sys.argv[2]
if len(sys.argv) >= 4:
    crop = sys.argv[3]

model_dir = "/data/deeplearning/zhangmm/segmentation/models/kddata_cls6_bjmnNet"
net = os.path.join(model_dir,"kddata_cls6_bjmnNet.prototxt")
weight = os.path.join(model_dir,"kddata_cls6_bjmnNet.caffemodel")


print("model: {}".format(net))
print("crop: {}".format(crop))

image_dir = "/data/deeplearning/zhangmm/segmentation/images/{}".format(dataset_name)
result_dir = os.path.join("/data/deeplearning/zhangmm/segmentation/output/caffe_result", os.path.basename(image_dir))
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
result_dir = os.path.join(result_dir, check_point)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

print("result dir: {}".format(result_dir))

# color map for result: 6 classes
colormap = np.asarray([
    [255, 0, 0], # tide lane, left_wait, guide_lane, in_out, and so on
    [255, 192, 203], # left lane, right lane
    [192, 128, 255], # bus lane
    [128, 64, 128], # road, fish lane, and so on
    [128, 0, 0], # objects on the road, fence
    [64, 64, 32], # sky, road signs, and so on
])

image_list = []
if os.path.isdir(image_dir):
    for fname in sorted(os.listdir(image_dir)):
        fpath = os.path.join(image_dir, fname)
        if fpath.endswith(".jpg") or fpath.endswith(".jpeg"):
            image_list.append(fpath)
        else:
            print("image file must be end with jpg/jpeg: {}".format(fpath))
else:
    image_list = image_dir

print("get images: {}".format(len(image_list)))

size = (1, 3, 512, 1224)
netcaffe = caffe.Net(net, weight, caffe.TEST)
tensor = np.ones(size, dtype=np.float32)


# read test image
batch_images = np.zeros((1, 3, 512, 1224), dtype=np.float32)
for i, image_path in enumerate(image_list):
    print("{}: {}".format(i, image_path))
    img = Image.open(image_path)
    if "true"==crop:
        print("crop: {}".format(image_path))
        w, h = img.size
        img = img.crop((0, h//2, w, h))
    img = np.array(img.resize((1224, 512))).transpose([2, 0, 1])
    #print("image shape: ".format(batch_images.shape))
    batch_images[0] = img
    #print(batch_images.shape)

    # predict
    netcaffe.blobs["data"].data[...] = batch_images
    netcaffe.forward()
    result = netcaffe.blobs[OUTPUT_NAME].data
    #result = np.argmax(result, axis=1)

    #print result
    # print("**** result: *****")
    # print(result)
    #print("**** result info: *****")
    #print("result shape: {}".format(result.shape))
    #print("result value: min={}; max={}".format(np.min(result),np.max(result)))

    result = result[0][0].astype(np.int8)
    #print("result shape: {}".format(result.shape))
    ## save to png image
    #color_image_path = os.path.join(result_dir, os.path.split(image_path)[1].split(".")[0] + ".jpg")
    color_image_path = os.path.join(result_dir, "frame_" + os.path.basename(image_path))
    
    color_result = colormap[result]
    color_image = Image.fromarray(color_result.astype(dtype=np.uint8))
    color_image.save(color_image_path, 'JPEG')



