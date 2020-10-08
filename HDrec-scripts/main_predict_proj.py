import tensorflow as tf
import numpy as np
import h5py, dxchange, argparse, os,shutil
from models import unet as make_generator_model           # import a generator model
from util import psnr


# Define parameters

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')


parser.add_argument('-depth', type=int, default=1, help='input depth')

parser.add_argument('-modelName', type=str, required=True, help='Trained model name')
parser.add_argument('-xtest', type=str, required=True, help='file name of X for testing')
parser.add_argument('-tomo', type=str, default="58", help='Sample number')

args, unparsed = parser.parse_known_args()


# Read parameters
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR
in_depth = args.depth



#generator = make_generator_model(input_shape=(None, None, in_depth))
#model_name = args.modelName
#generator.load_weights('./projection/model/%s.h5'%(model_name))#'./TomoGAN/Model/unet_proj_1000_19800.h5'
generator = tf.keras.models.load_model('./projection/model/%s.h5'%(args.modelName), )



# Load noisy images
noisyName = args.xtest
print('Loading noisy images')
hf_1 = h5py.File('./H5_Files/%s.h5'%(noisyName),'r')#'./H5_Files/sino_noisy_1000.h5'
data_noisy = hf_1['dataset'][:].astype(np.float32)
print(data_noisy.shape[0])





print('Predicting')
slice_num = range(data_noisy.shape[0])
print(len(slice_num))
data_recon = np.zeros((len(slice_num),data_noisy.shape[1],data_noisy.shape[2]))
#
img_test = np.exp(-data_noisy).reshape((data_noisy.shape[0],data_noisy.shape[1],data_noisy.shape[2],1))
print(img_test.shape)
img_rec = generator.predict(img_test,batch_size=2,verbose=1)
img_rec = img_rec.reshape((data_noisy.shape[0],data_noisy.shape[1],data_noisy.shape[2]))

img_rec = (img_rec - img_rec.min())/(img_rec.max()-img_rec.min())

img_rec = img_rec + img_rec.mean()
img_rec = img_rec /img_rec.max()

print('Predicted projection info')
min_data = img_rec.min()
max_data = img_rec.max()
print('Min value %f'%(min_data))
print('Max value %f'%(max_data ))


data_recon = -np.log(img_rec)
    
    

# for slice in slice_num:
#     img_test = np.exp(-data_noisy[slice,:,:]).reshape((1,data_noisy.shape[1],data_noisy.shape[2],1))
#     print(img_test.shape)
#     img_rec = generator.predict(img_test)
#     print('Slice number %d'%slice)
#     img_rec = img_rec.reshape((data_noisy.shape[1],data_noisy.shape[2]))
#     data_recon[slice,:,:] = -np.log(img_rec)


# data_recon = tp.remove_nan(data_recon, val=0.0)
# data_recon = tp.remove_neg(data_recon, val=0.00)
# data_recon[np.where(data_recon == np.inf)] = 0.00
# data_recon = tp.adjust_range(data_recon,np.percentile(data, (0.5)), np.percentile(data, (99.5)))
# data_recon = (data_recon - data_recon.min())/(data_recon.max()-data_recon.min())
    
print('Sinogram info')
print(data_recon.shape)
min_data = data_recon.min()
max_data = data_recon.max()
print('Min value %f'%(min_data))
print('Max value %f'%(max_data ))

# print('Min max normalization')
# data_recon = (data_recon - min_data)/(max_data - min_data)



hf = h5py.File('./result_%s/proj/H5_File/sino_recon_%s_proj.h5'%(args.tomo,args.modelName),'w')#'./H5_Files/sino_recon_1000.h5'
hf.create_dataset('dataset',data = data_recon)
hf.close()
    
    
hf_1.close()
    
