#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:07:00 2020

@author: zilingwu
"""

import tensorflow as tf
import numpy as np
import h5py, dxchange, argparse, os,shutil
from models import unet as make_generator_model           # import a generator model
from util import psnr
# from util import save2img
# Define parameters

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')

parser.add_argument('-depth', type=int, default=3, help='input depth')

parser.add_argument('-modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-proj_modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-sino_modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-dose', type=int, default=10,  help='Dose value')
parser.add_argument('-tomo', type=str, default="58", help='Sample number')

args, unparsed = parser.parse_known_args()


# Read parameters
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR
in_depth = args.depth

# Load models
print('Loading models')
generator = tf.keras.models.load_model('./projSino/model/%s.h5'%(args.modelName), )

# Load noisy images
print('Loading noisy images projSino/H5_File/sino_00058_combine_%d_%s_%s.h5'%(args.dose,\
     args.proj_modelName,args.sino_modelName))
hf_1 = h5py.File('./projSino/H5_File/sino_00058_combine_%d_%s_%s.h5'%(args.dose,\
     args.proj_modelName,args.sino_modelName), 'r')
data_noisy = hf_1['images'][:].astype(np.float32)
print(data_noisy.shape)
# save2img(data_noisy[0,:,:], './projSino/noisy_1.png')
# save2img(data_noisy[1,:,:], './projSino/noisy_2.png')
# save2img(data_noisy[2,:,:], './projSino/noisy_3.png')



# Prediction
print('Predicting')
slice_num = range(data_noisy.shape[0]//in_depth)
print(len(slice_num))
data_recon = np.zeros((len(slice_num),data_noisy.shape[1],data_noisy.shape[2]))



for slice in slice_num:
    img_test = np.exp(-data_noisy[slice*3:slice*3+3,:,:])
    img_test = np.transpose(img_test, (1, 2, 0))
    img_test = np.expand_dims(img_test, axis=0)
    #img_test = np.exp(-data_noisy[slice*3:slice*3+3,:,:]).reshape((1,data_noisy.shape[1],data_noisy.shape[2],3))
    print('Slice number %d'%slice)
    print(img_test.shape)
    img_rec = generator.predict(img_test)
    img_rec = img_rec.reshape((data_noisy.shape[1],data_noisy.shape[2]))
    data_recon[slice,:,:] = -np.log(img_rec)
    
# save2img(data_recon[0,:,:], './projSino/recon.png')   
hf = h5py.File('./result_%s/projSino/H5_File/sino_recon_%s_combine.h5'%(args.tomo,args.modelName),'w')#'./H5_Files/sino_recon_1000.h5'
hf.create_dataset('dataset',data = data_recon)
hf.close()



hf_1.close()










