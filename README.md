
# MXNet2Caffe: Convert MXNet model to Caffe model 

## UPDATE
1. 2019-01-08: Clone from https://github.com/GarrickLin/MXNet2Caffe.git
  - 增加log
  - 增加relu6(clip in mxnet): caffe中没有relu6，使用relu代替，之前的实验表明该替换对推断影响不大
  - 增加 Deconvolution：主要用于分割网络的最后几层
  - 增加 Upsampling：caffe中没有单独的上采样层，可以通过Deconvolution变相实现
  - 增加 SoftmaxOutput：该层在mxnet中为一层，转换到caffe后，使用两层：Softmax和Argmax


## USAGE
#### 使用环境
1. 必须同时安装caffe环境及mxnet环境，可以使用本人做好的docker镜像：docker pull kd-bd02.kuandeng.com/kd-recog/recog:mxnet2caffe
2. 修改脚本中mxnet的模型路径及caffe模型的保存路径，然后运行脚本即可：mxnet2caffe.sh

## TODO
1. 转换脚本中载入mxnet模型的方式不是太合理，但不影响使用
