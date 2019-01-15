
# MXNet2Caffe: Convert MXNet model to Caffe model 

## UPDATE
1. 2019-01-08: Clone from https://github.com/GarrickLin/MXNet2Caffe.git
  - ����log
  - ����relu6(clip in mxnet): caffe��û��relu6��ʹ��relu���棬֮ǰ��ʵ��������滻���ƶ�Ӱ�첻��
  - ���� Deconvolution����Ҫ���ڷָ��������󼸲�
  - ���� Upsampling��caffe��û�е������ϲ����㣬����ͨ��Deconvolution����ʵ��
  - ���� SoftmaxOutput���ò���mxnet��Ϊһ�㣬ת����caffe��ʹ�����㣺Softmax��Argmax


## USAGE
#### ʹ�û���
1. ����ͬʱ��װcaffe������mxnet����������ʹ�ñ������õ�docker����docker pull kd-bd02.kuandeng.com/kd-recog/recog:mxnet2caffe
2. �޸Ľű���mxnet��ģ��·����caffeģ�͵ı���·����Ȼ�����нű����ɣ�mxnet2caffe.sh

## TODO
1. ת���ű�������mxnetģ�͵ķ�ʽ����̫��������Ӱ��ʹ��
