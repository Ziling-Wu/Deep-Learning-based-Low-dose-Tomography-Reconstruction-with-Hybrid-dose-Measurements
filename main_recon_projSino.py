#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:07:30 2020

@author: zilingwu
"""



import tomopy as tp
import numpy as np
import dxchange as dx
import h5py, argparse, os,shutil
import tomorectv3d


def getp(a):
    return a.__array_interface__['data'][0]

# Define parameters

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-tomo', type=str, default="58", help='Sample number')

parser.add_argument('-modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-rotcenter', type=float, default = 1440.0,  help='rotation center')

args, unparsed = parser.parse_known_args()


# Read parameters
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR



# Load sinograms
print('Loading sinograms projSino/H5_File/sino_recon_%s_combine.h5'%(args.modelName))
hf_1 = h5py.File('./result_%s/projSino/H5_File/sino_recon_%s_combine.h5'%(args.tomo,args.modelName),'r')
#hf_1 = h5py.File('./H5_Files/%s.h5'%(sinoName),'r')#'./H5_Files/sino_noisy_1000.h5'
data = hf_1['dataset'][:].astype(np.float32)
print(data.shape)

# Refine sinogram
print('Refining sinograms')
data = tp.remove_nan(data, val=0.0)
data = tp.remove_neg(data, val=0.00)
data[np.where(data == np.inf)] = 0.00
data = tp.adjust_range(data,np.percentile(data, (0.5)), np.percentile(data, (99.5)))
data = (data - data.min())/(data.max()-data.min())

# generate folders to save results
print('Generating folders')
itr_out_dir_proj =  './result_%s/projSino/projection/sino_recon_%s_combine_new'%(args.tomo, args.modelName)
if os.path.isdir(itr_out_dir_proj): 
    shutil.rmtree(itr_out_dir_proj)
os.mkdir(itr_out_dir_proj) # to save temp output

itr_out_dir_sino =  './result_%s/projSino/sinogram/sino_recon_%s_combine_new'%(args.tomo,args.modelName)
if os.path.isdir(itr_out_dir_sino): 
    shutil.rmtree(itr_out_dir_sino)
os.mkdir(itr_out_dir_sino) # to save temp output


itr_out_dir_recon_tv =  './result_%s/projSino/reconstruction/TV/sino_recon_%s_combine_new'%(args.tomo,args.modelName)
if os.path.isdir(itr_out_dir_recon_tv): 
    shutil.rmtree(itr_out_dir_recon_tv)
os.mkdir(itr_out_dir_recon_tv) # to save temp output

itr_out_dir_recon_gridrec =  './result_%s/projSino/reconstruction/gridrec/sino_recon_%s_combine_new'%(args.tomo,args.modelName)
if os.path.isdir(itr_out_dir_recon_gridrec): 
    shutil.rmtree(itr_out_dir_recon_gridrec)
os.mkdir(itr_out_dir_recon_gridrec) # to save temp output


# Save sinogram images
print('Saving sinogram images')   #2160 sinograms
slice_num = [400,700,1000,1300,1600,1900,2100]#, 1479] 
  
# Save sinogram
for slice in slice_num:
    print('Slice number %d'%slice)
    img_test = data[:,slice,:].reshape((1,data.shape[0],data.shape[2],1))
    fname_save = '%s/shale_sino_%d'%(itr_out_dir_sino,slice)
    dx.write_tiff(img_test, fname_save, dtype='float32')
    
    

# Save projection images
print('Saving projection images')   # 1501 projections
slice_num = [55,200,500,805,1100,1315, 1400]#, 1479] 
  
# Save projection
for slice in slice_num:
    print('Slice number %d'%slice)
    img_test = np.exp(-data[slice,:,:]).reshape((1,data.shape[1],data.shape[2],1))
    fname_save = '%s/shale_proj_%d'%(itr_out_dir_proj,slice)
    dx.write_tiff(img_test, fname_save, dtype='float32')
    
# save reconstruction
print("Starting reconstruction") #1792 reconstructions
theta = tp.angles(data.shape[0])
print(theta.shape)
rot_center = args.rotcenter


N = data.shape[2]
Ntheta = data.shape[0]
Nz = 1
Nzp = 1  # number of slices for simultaneous processing by 1 gpu
ngpus = 1  # number of gpus to process the data (index 0,1,2,.. are taken)
niter = 256  # number of iterations in the Chambolle-Pock algorithm
method = 0 # 0:tv, 1:tv entropy, 2:tv L1
lambda0 = 1e-5

slice_num = [400,700,1000,1300,1600,1900,2100]#, 1479] 
cl = tomorectv3d.tomorectv3d(N, Ntheta, Nz, Nzp, ngpus, rot_center, lambda0)
theta = np.array(np.linspace(
    0, np.pi, Ntheta).astype('float32'))
cl.settheta(getp(theta))
for slice in slice_num:
    print('Slice number %d'%slice)
    start = slice
    end = slice+1
    data_rec = data[:,start:end,:]
    data_rec_slice = np.array(data_rec.swapaxes(0,1))
    print(data_rec.shape)
    print(data_rec_slice.shape)
    print(data_rec_slice.dtype)
    
    # generate and set angles
    
    # reconstruction with 3d tv
    res = np.zeros([Nz, N, N], dtype='float32')
    
    cl.itertvR_wrap(getp(res), getp(data_rec_slice), niter)
    # Mask each reconstructed slice with a circle so that high/low pixel values are removed.
    recon = tp.circ_mask(res, axis=0, ratio=0.95)
    print("Writing output")
    dx.write_tiff(recon, fname= '%s/shale_rec_'%(itr_out_dir_recon_tv)+str(slice)+'.tiff', overwrite=True)
    print('Slice number %d reconstructed'%slice)
    
    
    # reconstruction with gridrec
    rec = tp.recon(data_rec, theta, center=rot_center, algorithm='gridrec',filter_name = 'parzen')#,num_iter=1000,reg_par = 0.01

    # Mask each reconstructed slice with a circle so that high/low pixel values are removed.
    # rec = tp.remove_ring(rec, center_x=1024, center_y=1024)

    # Mask each reconstructed slice with a circle so that high/low pixel values are removed.
    recon = tp.circ_mask(rec, axis=0, ratio=0.95)

    print("Writing output")
    dx.write_tiff_stack(recon, fname= '%s/shale_rec'%(itr_out_dir_recon_gridrec), start=start)

hf_1.close()
    
