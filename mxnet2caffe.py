import sys, argparse
import mxnet as mx
import caffe

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',    type=str, default='model_mxnet/face/facega2')
parser.add_argument('--mx-epoch',    type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/face/facega2.prototxt')
parser.add_argument('--cf-model',    type=str, default='model_caffe/face/facega2.caffemodel')
args = parser.parse_args()

# ------------------------------------------
# Load
_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
#net = caffe.Net(args.cf_prototxt, caffe.TRAIN)   
net = caffe.Net(args.cf_prototxt, caffe.TEST)   


# ------------------------------------------
# Convert
all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()

print('----------------------------------\n')
print('ALL KEYS IN MXNET:')
print(all_keys)
print('%d KEYS' %len(all_keys))


print('----------------------------------\n')
print('VALID KEYS:')
for i_key,key_i in enumerate(all_keys):

  try:    
    
    if 'data' is key_i:
      pass
    elif '_weight' in key_i:
      key_caffe = key_i.replace('_weight','')
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
      # if 'fc' in key_i:
        # print key_i
        # print arg_params[key_i].shape
        # print net.params[key_caffe][0].data.shape
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_bias' in key_i:
      key_caffe = key_i.replace('_bias','')
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat   
    elif '_gamma' in key_i and 'relu' not in key_i:
      key_caffe = key_i.replace('_gamma','_scale')
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    # TODO: support prelu
    elif '_gamma' in key_i and 'relu' in key_i:   # for prelu
      key_caffe = key_i.replace('_gamma','')
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
      assert (len(net.params[key_caffe]) == 1)
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_beta' in key_i:
      key_caffe = key_i.replace('_beta','_scale')
      print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat
    elif '_moving_mean' in key_i:
      key_caffe = key_i.replace('_moving_mean','')
      print("{}: {}->{}".format(key_i, aux_params[key_i].shape, net.params[key_caffe][0].data.shape))
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat 
      net.params[key_caffe][2].data[...] = 1
    elif '_moving_var' in key_i:
      key_caffe = key_i.replace('_moving_var','')
      print("{}: {}->{}".format(key_i, aux_params[key_i].shape, net.params[key_caffe][0].data.shape))
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat    
      net.params[key_caffe][2].data[...] = 1
    else:
      sys.exit("Warning!  Unknown mxnet:{}".format(key_i))
  
    print("% 3d | %s -> %s, initialized." 
           %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    
  except KeyError:
    import traceback
    print(traceback.print_exc())
    print("\nError!  key error mxnet:{}".format(key_i))  
    break
      
# ------------------------------------------
# Finish
net.save(args.cf_model)

print("\n*** PARAMS to CAFFEMODEL Finished. ***\n")



