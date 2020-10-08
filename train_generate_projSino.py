#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:18:04 2020

@author: zilingwu
"""


import tensorflow as tf
import numpy as np
import h5py, argparse, os
from util import save2img
parser = argparse.ArgumentParser(description='Generate training datasets for two-step training')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-dose', type=int, default=10,  help='Dose value')
parser.add_argument('-proj_modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-sino_modelName', type=str, required=True, help='Trained model name')
args, unparsed = parser.parse_known_args()

# Read parameters
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR





# projection
print('Original noisy image reading')


hf_1 = h5py.File('./projection/dataset/noisy_train_128_%d.h5'%(args.dose), 'r')
# data_noisy = hf_1.get('images')
data_noisy = hf_1['images'][:].astype(np.float32)
print(data_noisy.shape)
print(data_noisy.min())
print(data_noisy.max())



# prediction
print('loading prediction models in projection domain')
 
generator = tf.keras.models.load_model('./projection/model/%s.h5'%(args.proj_modelName), )

print('Predicting in projection domain')
slice_num = range(data_noisy.shape[0])
print(len(slice_num))
data_recon = np.zeros((len(slice_num),data_noisy.shape[1],data_noisy.shape[2]))


for slice in slice_num:
    img_test = (data_noisy[slice,:,:]).reshape((1,data_noisy.shape[1],data_noisy.shape[2],1))
    #print(img_test.shape)
    img_rec = generator.predict(img_test)
    #print('Slice number %d'%slice)
    img_rec = img_rec.reshape((data_noisy.shape[1],data_noisy.shape[2]))
    data_recon[slice,:,:] = (img_rec)
print(data_recon.shape)    

save2img(data_recon[0,:,:], './projSino/proj.png')    

print('loading prediction models in sinogram domain')
 
generator = tf.keras.models.load_model('./sinogram/model/%s.h5'%(args.sino_modelName), )

print('Predicting in sinogram domain')  


x_test = np.asarray(data_noisy, dtype=np.float32)
x_test = np.transpose(x_test, (1, 0, 2))  # (1792,128,2048)
test = (x_test).reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
img_rec = generator.predict(test)
img_rec = img_rec.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2]))
img_rec = np.transpose(img_rec, (1, 0, 2)) 
print(img_rec.shape)
save2img(img_rec[0,:,:], './projSino/sino.png')   
save2img(data_noisy[0,:,:], './projSino/noisy.png')     

# Combination as noisy input
arrays = []
print('Combination')
for i in range(data_recon.shape[0]):
    arrays.append( data_noisy[i])
    arrays.append( data_recon[i])
    arrays.append( img_rec[i])
arrays = np.asarray(arrays, dtype=np.float32)
print(arrays.shape)

#
print('Saving noisy dataset into projSino/dataset')
hf = h5py.File('./projSino/dataset/noisy_train_%d.h5'%(args.dose), 'w')
hf.create_dataset('images', data = arrays)
hf.close()

print('Original clean image reading')

hf_2 = h5py.File('./projection/dataset/clean_train_128.h5', 'r')
data = hf_2.get('images')
print(data.shape)

print('Saving clean dataset into projSino/dataset')

hf = h5py.File('./projSino/dataset/clean_train.h5', 'w')
hf.create_dataset('images', data = data)
hf.close()

save2img(arrays[0,:,:], './projSino/noisy_1.png')
save2img(arrays[1,:,:], './projSino/noisy_2.png')
save2img(arrays[2,:,:], './projSino/noisy_3.png')

save2img(data[0,:,:], './projSino/clean.png')



print('Saving test noisy dataset into projSino/dataset')
arrays_test = arrays[0:6]
hf = h5py.File('./projSino/dataset/noisy_test_%d.h5'%(args.dose), 'w')
hf.create_dataset('images', data = arrays_test)
hf.close()

data_test = data[0:2]

hf = h5py.File('./projSino/dataset/clean_test.h5', 'w')
hf.create_dataset('images', data = data_test)
hf.close()
hf_2.close()



hf_1.close()
