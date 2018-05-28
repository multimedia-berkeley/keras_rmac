from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
from keras.utils import multi_gpu_model
import keras.backend as K

from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

import scipy.io
import numpy as np
import utils
import time


#K.set_image_dim_ordering('th')
def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):

    # Load VGG16
   # vgg16_model = VGG16(utils.DATA_DIR + utils.WEIGHTS_FILE, input_shape)
    vgg16_model = VGG16(input_shape=input_shape, include_top=False)

    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-1].output, in_roi])
 #   print('ROI pooling layer name:', vgg16_model.layers[-1].output)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model(inputs=[vgg16_model.input, in_roi], outputs=rmac_norm)

    # Load PCA weights
    mat = scipy.io.loadmat(utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])
 #   print('layer name:', model.layers[-4].name)
  #  for idx, layer in enumerate(model.layers):
  #      print(idx, layer.name)

    return model


if __name__ == "__main__":

    # Load sample image
    file = utils.DATA_DIR + 'sample.jpg'
    
    img = image.load_img(file)

    # Resize
    scale = utils.IMG_SIZE / max(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
    img = img.resize(new_size)
    
    num_data = 128
    print('num_data:', num_data)
    # Mean substraction
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
#    x = np.array([x,x])
    #x = np.repeat(x, num_data, axis=0)
    x = np.array([x for i in range(num_data)])
    print(x.shape)

    #x = np.array([x,x])
    x = utils.preprocess_image(x)
    

    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1]) #image_dim_ordering tf
    #Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2]) #image_dim_ordering th
    regions = rmac_regions(Wmap, Hmap, 3)
    
    print('Loading RMAC model...')
    print(x.shape[1], x.shape[2], x.shape[3], len(regions))
    
    PARALLEL = True
    num_gpu = 2
    if PARALLEL:
        with tf.device('/cpu:0'):
            model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
        parallel_model = multi_gpu_model(model, gpus=num_gpu)
    
    else:
        model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
     

    #regions = np.array([regions, regions])
    # Compute RMAC vector
    print('Extracting RMAC from image...')
    #regions = np.expand_dims(regions, axis=0) 
    #regions = np.array([regions,regions])
    #regions = np.tile(regions, (num_data, 1))
    regions = np.array([regions for i in range(num_data)])
    print(np.shape(regions))
    print(np.shape(x))
    #input_x = [np.array([x,x]), np.array([regions, regions])]
    input_x = [x, regions] 
    start = time.time()
    if PARALLEL:
        RMAC = parallel_model.predict(input_x, batch_size=8 * num_gpu)
    else:
        RMAC = model.predict(input_x, batch_size=8)
    print('RMAC size:', RMAC.shape)
    print(time.time() - start, 'seconds for %d images'%(num_data))

    #print(RMAC)
    #print(sorted(RMAC[0]))
    print('norm:', np.linalg.norm(RMAC[0]))
    print('Done!')
