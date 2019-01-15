#/bin/bash

root_model_dir="/data/deeplearning/zhangmm/segmentation/models/kddata_cls6_bjmnNet"
mx_json="${root_model_dir}/kddata_cls6_bjmnNet_ep-symbol.json"
mx_model="${root_model_dir}/kddata_cls6_bjmnNet_ep"
cf_prototxt="${root_model_dir}/kddata_cls6_bjmnNet.prototxt" 
cf_model="${root_model_dir}/kddata_cls6_bjmnNet.caffemodel"
epoch=300

echo "json 2 prototxt..."
python json2prototxt.py --mx-json ${mx_json} --cf-prototxt ${cf_prototxt}

echo "param 2 caffe"
python mxnet2caffe.py --mx-model ${mx_model} --mx-epoch ${epoch} --cf-prototxt ${cf_prototxt} --cf-model ${cf_model}
