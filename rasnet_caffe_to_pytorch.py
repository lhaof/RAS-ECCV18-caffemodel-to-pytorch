from __future__ import division
import sys
caffe_root = '../caffe_dss-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from rasnet_naive import rasnet, interp_surgery
from PIL import Image

caffe.set_mode_gpu()
caffe.set_device(0)
caffeproto = 'deploy.prototxt'
caffeweight = './snapshot/ras_iter_10000.caffemodel'
caffenet = caffe.Net(caffeproto, caffeweight, caffe.TEST)

pthnet = rasnet()
pthnet.cuda()
pthnet = interp_surgery(pthnet) 

print caffenet.params.keys()
print pthnet.state_dict().keys()

pthnet.state_dict()['conv1_1.weight'].copy_(torch.from_numpy(caffenet.params['conv1_1'][0].data))
pthnet.state_dict()['conv1_1.bias'].copy_(torch.from_numpy(caffenet.params['conv1_1'][1].data))
pthnet.state_dict()['conv1_2.weight'].copy_(torch.from_numpy(caffenet.params['conv1_2'][0].data))
pthnet.state_dict()['conv1_2.bias'].copy_(torch.from_numpy(caffenet.params['conv1_2'][1].data))

pthnet.state_dict()['conv2_1.weight'].copy_(torch.from_numpy(caffenet.params['conv2_1'][0].data))
pthnet.state_dict()['conv2_1.bias'].copy_(torch.from_numpy(caffenet.params['conv2_1'][1].data))
pthnet.state_dict()['conv2_2.weight'].copy_(torch.from_numpy(caffenet.params['conv2_2'][0].data))
pthnet.state_dict()['conv2_2.bias'].copy_(torch.from_numpy(caffenet.params['conv2_2'][1].data))

pthnet.state_dict()['conv3_1.weight'].copy_(torch.from_numpy(caffenet.params['conv3_1'][0].data))
pthnet.state_dict()['conv3_1.bias'].copy_(torch.from_numpy(caffenet.params['conv3_1'][1].data))
pthnet.state_dict()['conv3_2.weight'].copy_(torch.from_numpy(caffenet.params['conv3_2'][0].data))
pthnet.state_dict()['conv3_2.bias'].copy_(torch.from_numpy(caffenet.params['conv3_2'][1].data))
pthnet.state_dict()['conv3_3.weight'].copy_(torch.from_numpy(caffenet.params['conv3_3'][0].data))
pthnet.state_dict()['conv3_3.bias'].copy_(torch.from_numpy(caffenet.params['conv3_3'][1].data))

pthnet.state_dict()['conv4_1.weight'].copy_(torch.from_numpy(caffenet.params['conv4_1'][0].data))
pthnet.state_dict()['conv4_1.bias'].copy_(torch.from_numpy(caffenet.params['conv4_1'][1].data))
pthnet.state_dict()['conv4_2.weight'].copy_(torch.from_numpy(caffenet.params['conv4_2'][0].data))
pthnet.state_dict()['conv4_2.bias'].copy_(torch.from_numpy(caffenet.params['conv4_2'][1].data))
pthnet.state_dict()['conv4_3.weight'].copy_(torch.from_numpy(caffenet.params['conv4_3'][0].data))
pthnet.state_dict()['conv4_3.bias'].copy_(torch.from_numpy(caffenet.params['conv4_3'][1].data))

pthnet.state_dict()['conv5_1.weight'].copy_(torch.from_numpy(caffenet.params['conv5_1'][0].data))
pthnet.state_dict()['conv5_1.bias'].copy_(torch.from_numpy(caffenet.params['conv5_1'][1].data))
pthnet.state_dict()['conv5_2.weight'].copy_(torch.from_numpy(caffenet.params['conv5_2'][0].data))
pthnet.state_dict()['conv5_2.bias'].copy_(torch.from_numpy(caffenet.params['conv5_2'][1].data))
pthnet.state_dict()['conv5_3.weight'].copy_(torch.from_numpy(caffenet.params['conv5_3'][0].data))
pthnet.state_dict()['conv5_3.bias'].copy_(torch.from_numpy(caffenet.params['conv5_3'][1].data))

pthnet.state_dict()['conv1_dsn6.weight'].copy_(torch.from_numpy(caffenet.params['conv1-dsn6'][0].data))
pthnet.state_dict()['conv1_dsn6.bias'].copy_(torch.from_numpy(caffenet.params['conv1-dsn6'][1].data))
pthnet.state_dict()['conv2_dsn6.weight'].copy_(torch.from_numpy(caffenet.params['conv2-dsn6'][0].data))
pthnet.state_dict()['conv2_dsn6.bias'].copy_(torch.from_numpy(caffenet.params['conv2-dsn6'][1].data))
pthnet.state_dict()['conv3_dsn6.weight'].copy_(torch.from_numpy(caffenet.params['conv3-dsn6'][0].data))
pthnet.state_dict()['conv3_dsn6.bias'].copy_(torch.from_numpy(caffenet.params['conv3-dsn6'][1].data))
pthnet.state_dict()['conv4_dsn6.weight'].copy_(torch.from_numpy(caffenet.params['conv4-dsn6'][0].data))
pthnet.state_dict()['conv4_dsn6.bias'].copy_(torch.from_numpy(caffenet.params['conv4-dsn6'][1].data))
pthnet.state_dict()['conv5_dsn6.weight'].copy_(torch.from_numpy(caffenet.params['conv5-dsn6'][0].data))
pthnet.state_dict()['conv5_dsn6.bias'].copy_(torch.from_numpy(caffenet.params['conv5-dsn6'][1].data))

pthnet.state_dict()['conv1_dsn5.weight'].copy_(torch.from_numpy(caffenet.params['conv1-dsn5'][0].data))
pthnet.state_dict()['conv1_dsn5.bias'].copy_(torch.from_numpy(caffenet.params['conv1-dsn5'][1].data))
pthnet.state_dict()['conv2_dsn5.weight'].copy_(torch.from_numpy(caffenet.params['conv2-dsn5'][0].data))
pthnet.state_dict()['conv2_dsn5.bias'].copy_(torch.from_numpy(caffenet.params['conv2-dsn5'][1].data))
pthnet.state_dict()['conv3_dsn5.weight'].copy_(torch.from_numpy(caffenet.params['conv3-dsn5'][0].data))
pthnet.state_dict()['conv3_dsn5.bias'].copy_(torch.from_numpy(caffenet.params['conv3-dsn5'][1].data))
pthnet.state_dict()['conv4_dsn5.weight'].copy_(torch.from_numpy(caffenet.params['conv4-dsn5'][0].data))
pthnet.state_dict()['conv4_dsn5.bias'].copy_(torch.from_numpy(caffenet.params['conv4-dsn5'][1].data))

pthnet.state_dict()['conv1_dsn4.weight'].copy_(torch.from_numpy(caffenet.params['conv1-dsn4'][0].data))
pthnet.state_dict()['conv1_dsn4.bias'].copy_(torch.from_numpy(caffenet.params['conv1-dsn4'][1].data))
pthnet.state_dict()['conv2_dsn4.weight'].copy_(torch.from_numpy(caffenet.params['conv2-dsn4'][0].data))
pthnet.state_dict()['conv2_dsn4.bias'].copy_(torch.from_numpy(caffenet.params['conv2-dsn4'][1].data))
pthnet.state_dict()['conv3_dsn4.weight'].copy_(torch.from_numpy(caffenet.params['conv3-dsn4'][0].data))
pthnet.state_dict()['conv3_dsn4.bias'].copy_(torch.from_numpy(caffenet.params['conv3-dsn4'][1].data))
pthnet.state_dict()['conv4_dsn4.weight'].copy_(torch.from_numpy(caffenet.params['conv4-dsn4'][0].data))
pthnet.state_dict()['conv4_dsn4.bias'].copy_(torch.from_numpy(caffenet.params['conv4-dsn4'][1].data))

pthnet.state_dict()['conv1_dsn3.weight'].copy_(torch.from_numpy(caffenet.params['conv1-dsn3'][0].data))
pthnet.state_dict()['conv1_dsn3.bias'].copy_(torch.from_numpy(caffenet.params['conv1-dsn3'][1].data))
pthnet.state_dict()['conv2_dsn3.weight'].copy_(torch.from_numpy(caffenet.params['conv2-dsn3'][0].data))
pthnet.state_dict()['conv2_dsn3.bias'].copy_(torch.from_numpy(caffenet.params['conv2-dsn3'][1].data))
pthnet.state_dict()['conv3_dsn3.weight'].copy_(torch.from_numpy(caffenet.params['conv3-dsn3'][0].data))
pthnet.state_dict()['conv3_dsn3.bias'].copy_(torch.from_numpy(caffenet.params['conv3-dsn3'][1].data))
pthnet.state_dict()['conv4_dsn3.weight'].copy_(torch.from_numpy(caffenet.params['conv4-dsn3'][0].data))
pthnet.state_dict()['conv4_dsn3.bias'].copy_(torch.from_numpy(caffenet.params['conv4-dsn3'][1].data))

pthnet.state_dict()['conv1_dsn2.weight'].copy_(torch.from_numpy(caffenet.params['conv1-dsn2'][0].data))
pthnet.state_dict()['conv1_dsn2.bias'].copy_(torch.from_numpy(caffenet.params['conv1-dsn2'][1].data))
pthnet.state_dict()['conv2_dsn2.weight'].copy_(torch.from_numpy(caffenet.params['conv2-dsn2'][0].data))
pthnet.state_dict()['conv2_dsn2.bias'].copy_(torch.from_numpy(caffenet.params['conv2-dsn2'][1].data))
pthnet.state_dict()['conv3_dsn2.weight'].copy_(torch.from_numpy(caffenet.params['conv3-dsn2'][0].data))
pthnet.state_dict()['conv3_dsn2.bias'].copy_(torch.from_numpy(caffenet.params['conv3-dsn2'][1].data))
pthnet.state_dict()['conv4_dsn2.weight'].copy_(torch.from_numpy(caffenet.params['conv4-dsn2'][0].data))
pthnet.state_dict()['conv4_dsn2.bias'].copy_(torch.from_numpy(caffenet.params['conv4-dsn2'][1].data))

pthnet.state_dict()['conv1_dsn1.weight'].copy_(torch.from_numpy(caffenet.params['conv1-dsn1'][0].data))
pthnet.state_dict()['conv1_dsn1.bias'].copy_(torch.from_numpy(caffenet.params['conv1-dsn1'][1].data))
pthnet.state_dict()['conv2_dsn1.weight'].copy_(torch.from_numpy(caffenet.params['conv2-dsn1'][0].data))
pthnet.state_dict()['conv2_dsn1.bias'].copy_(torch.from_numpy(caffenet.params['conv2-dsn1'][1].data))
pthnet.state_dict()['conv3_dsn1.weight'].copy_(torch.from_numpy(caffenet.params['conv3-dsn1'][0].data))
pthnet.state_dict()['conv3_dsn1.bias'].copy_(torch.from_numpy(caffenet.params['conv3-dsn1'][1].data))
pthnet.state_dict()['conv4_dsn1.weight'].copy_(torch.from_numpy(caffenet.params['conv4-dsn1'][0].data))
pthnet.state_dict()['conv4_dsn1.bias'].copy_(torch.from_numpy(caffenet.params['conv4-dsn1'][1].data))

pthnet.state_dict()['upsample32_dsn6.weight'].copy_(torch.from_numpy(caffenet.params['upsample32_dsn6'][0].data))
pthnet.state_dict()['upsample32_dsn6.bias'].copy_(torch.from_numpy(caffenet.params['upsample32_dsn6'][1].data))
pthnet.state_dict()['upsample16_dsn5.weight'].copy_(torch.from_numpy(caffenet.params['upsample16_dsn5'][0].data))
pthnet.state_dict()['upsample16_dsn5.bias'].copy_(torch.from_numpy(caffenet.params['upsample16_dsn5'][1].data))
pthnet.state_dict()['upsample8_dsn4.weight'].copy_(torch.from_numpy(caffenet.params['upsample8_dsn4'][0].data))
pthnet.state_dict()['upsample8_dsn4.bias'].copy_(torch.from_numpy(caffenet.params['upsample8_dsn4'][1].data))
pthnet.state_dict()['upsample4_dsn3.weight'].copy_(torch.from_numpy(caffenet.params['upsample4_dsn3'][0].data))
pthnet.state_dict()['upsample4_dsn3.bias'].copy_(torch.from_numpy(caffenet.params['upsample4_dsn3'][1].data))
pthnet.state_dict()['upsample2_1_dsn2.weight'].copy_(torch.from_numpy(caffenet.params['upsample2_1-dsn2'][0].data))
pthnet.state_dict()['upsample2_1_dsn2.bias'].copy_(torch.from_numpy(caffenet.params['upsample2_1-dsn2'][1].data))
pthnet.state_dict()['upsample2_2_dsn2.weight'].copy_(torch.from_numpy(caffenet.params['upsample2_2-dsn2'][0].data))
pthnet.state_dict()['upsample2_2_dsn2.bias'].copy_(torch.from_numpy(caffenet.params['upsample2_2-dsn2'][1].data))
	
imname = './dataset/HKU-IS/imgs/0004.png'
im = Image.open(imname).convert('RGB')
#im = im.resize((224,224)) 
im = np.array(im).astype(np.float32) 
im = im[:, :, ::-1]
im -= np.array((104.00699, 116.66877, 122.67892))
im = im.transpose((2,0,1))
im = np.ascontiguousarray(im) 
im = np.expand_dims(im, axis=0) 

caffenet.blobs['data'].reshape(*im.shape) 
caffenet.blobs['data'].data[...] = im 
caffenet.forward()
caffeout = caffenet.blobs['sigmoid-score1'].data.copy()
print caffeout.shape, caffeout.min(), caffeout.mean(), caffeout.max()

pthnet.eval()
pthout = pthnet.forward(torch.from_numpy(im).cuda())
pthout = pthout.cpu().data.numpy()
print pthout.shape, pthout.min(), pthout.mean(), pthout.max()
print 'done'