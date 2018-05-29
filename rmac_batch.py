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
import os
import cPickle as pickle
import sys
import traceback

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



def load_model(x, multi='parallel'):
    x = np.array(x)
    print('cur batch tensor shape', x.shape)
    x = utils.preprocess_image(x)

    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[2], x.shape[1]) #image_dim_ordering tf
    #Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2]) #image_dim_ordering th
    regions = rmac_regions(Wmap, Hmap, 3)
    
    print('Loading RMAC model...')
    print(x.shape[1], x.shape[2], x.shape[3], len(regions))
    
    if PARALLEL:
        if multi=='single':
            model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
            return model, regions
        else:
            with tf.device('/cpu:0'):
                model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
            parallel_model = multi_gpu_model(model, gpus=num_gpu)
            return parallel_model, regions
    else:
        model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
        return model, regions
    

def extract_feature(x, model, regions):
    num_data = len(x)
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
        RMAC = model.predict(input_x, batch_size=BATCH_SIZE * num_gpu)
    else:
        RMAC = model.predict(input_x, batch_size=BATCH_SIZE)
    print('RMAC size:', RMAC.shape)
    print(time.time() - start, 'seconds for %d images'%(num_data))
    #print(RMAC)
    #print(sorted(RMAC[0]))
    #print('norm:', np.linalg.norm(RMAC[0]))
    return RMAC

if __name__ == "__main__":
    INPUT_FILE = sys.argv[1]
    LAST_MINUTE = True
    split_name = INPUT_FILE.split('_')[1]
    print('Split:', split_name)
    PARALLEL = True
    num_gpu = 16#pascal 2
    BATCH_SIZE = 24#pascal 40
    # Load sample image
#    file = utils.DATA_DIR + 'sample.jpg'
    DATASET = 'test' 
    output_file = 'rmac_' + DATASET + '_' + split_name + '.pickle'
    output_file = os.path.join('/g/g92/choi13/src/keras_rmac/rmac_result', output_file)
    print('output_file name', output_file)
    try:
        with open(output_file, 'rb') as f:
            partial_result = pickle.load(f)
            filename_output = partial_result[0]
            rmac_result = partial_result[1]
            assert len(rmac_result[0]) == len(rmac_result[1])
            print('SKipping', len(filename_output), 'files')
            completed_files = set(filename_output)
    except:
        print(split_name)
        print(traceback.print_exc())
        rmac_result = list()
        filename_output = list()
    
    if LAST_MINUTE: 
        rmac_result = list()
        filename_output = list()

    #PATH_IMAGE = '/g/g92/choi13/projects/landmark/data/recognition/' + DATASET + '_resized'
    PATH_IMAGE = '/data/landmark/images/' + DATASET + '_resized'
    with open('../landmark/'+ INPUT_FILE, 'rb') as f:
        completed_files = list()

    if LAST_MINUTE:
        filename_output = list()
        rmac_result = list()

    #PATH_IMAGE = '/data/landmark/images/' + DATASET + '_resized'
    with open('/g/g92/choi13/projects/landmark/'+ INPUT_FILE, 'rb') as f:
        d = pickle.load(f)

    #for key in d.keys():
    len_by_key = [(key, len(d[key])) for key in d.keys()]
    len_by_key = sorted(len_by_key, key=lambda x:x[1], reverse=True)
    count = 0 
    for size, _ in len_by_key:
        filelist = d[size]
        #for filename in filelist:
        first_batch = True
        start = time.time()
        while len(filelist) > 0:
            cur_batch = filelist[:BATCH_SIZE * num_gpu]
            filelist = filelist[BATCH_SIZE * num_gpu:]
            l_imgs = list()
            for filename in cur_batch:
                if filename in completed_files:
                    continue
                cur_file = os.path.join(PATH_IMAGE, filename)
                img = image.load_img(cur_file)
                x = image.img_to_array(img)
                l_imgs.append(x)
            l_imgs = np.array(l_imgs)
            #print('cur batch tensor shape', l_imgs.shape)
            if l_imgs.shape[0] == 0 :
                continue
        
            filename_output.extend(cur_batch) 
            l_imgs = utils.preprocess_image(l_imgs)
            if len(l_imgs) < num_gpu:
                model, regions = load_model(l_imgs, 'single')

            elif first_batch: 
                model, regions = load_model(l_imgs)
                first_batch = False 

            rmac_batch = extract_feature(l_imgs, model, regions)
            rmac_result.extend(rmac_batch)
            print('[%s]'%(split_name), len(filelist), 'remaining. finished:', len(rmac_result), (time.time() - start)/(BATCH_SIZE* num_gpu) * 100, 'seconds for 100 images')
            start = time.time()
            count += len(rmac_batch) 
            if count > 200:
                count = 0 
                with open(output_file, 'wb') as f:
                    pickle.dump((filename_output, rmac_result), f)
                print('Saved rmac_result')
                    

    with open(output_file, 'wb') as f:
        pickle.dump((filename_output, rmac_result), f)
    # Resize
    #scale = utils.IMG_SIZE / max(img.size)
    #new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    #print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))
    #img = img.resize(new_size)
    
    #num_data = num_gpu * BATCH_SIZE
    # Mean substraction
    
    
    #x = np.expand_dims(x, axis=0)
#    x = np.array([x,x])
    #x = np.repeat(x, num_data, axis=0)
    #x = np.array([x for i in range(num_data)])
    #print(x.shape)

    #x = np.array([x,x])
