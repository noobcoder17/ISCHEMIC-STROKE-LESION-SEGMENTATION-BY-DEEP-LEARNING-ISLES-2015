from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

ign = np.load('./data/normal_data/training/imageNames.npy')
imagenames = []
for i in range(ign.shape[0]):
    imagenames.append(ign[i][0])

random.shuffle(imagenames)
print('total images={}'.format(len(imagenames)))
split = int(len(imagenames)*0.8)

trainImageNames = imagenames[:split]
validImageNames = imagenames[split:]
print('training images={}'.format(len(trainImageNames)))
print('validation images={}'.format(len(validImageNames)))

print(trainImageNames[:5])
print(validImageNames[:5])

#augmented image from one image
ifi = 7

data_gen_args = dict(
                    rescale=1.0/255,
                    rotation_range=30,
                    horizontal_flip=True,
                    vertical_flip=True,
                    shear_range=0.2,
                    zoom_range=0.1)

def trainset(b_size):
    print('creating augmented training images...')
    seed = 1337
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    save_here_cbf = 'data/augmented_data/training/cbf'
    save_here_cbv = 'data/augmented_data/training/cbv'
    save_here_dwi = 'data/augmented_data/training/dwi'
    save_here_t1c = 'data/augmented_data/training/t1c'
    save_here_t2 = 'data/augmented_data/training/t2'
    save_here_tmax = 'data/augmented_data/training/tmax'
    save_here_ttp = 'data/augmented_data/training/ttp'
    save_here_mask = 'data/augmented_data/training/mask'

    k=0
    for i in range(len(trainImageNames)):
        normalcbfPath = 'data/normal_data/training/cbf/{}'.format(trainImageNames[i])
        normalcbvPath = 'data/normal_data/training/cbv/{}'.format(trainImageNames[i])
        normaldwiPath = 'data/normal_data/training/dwi/{}'.format(trainImageNames[i])
        normalt1cPath = 'data/normal_data/training/t1c/{}'.format(trainImageNames[i])
        normalt2Path = 'data/normal_data/training/t2/{}'.format(trainImageNames[i])
        normaltmaxPath = 'data/normal_data/training/tmax/{}'.format(trainImageNames[i])
        normalttpPath = 'data/normal_data/training/ttp/{}'.format(trainImageNames[i])
        normalmaskPath = 'data/normal_data/training/mask/{}'.format(trainImageNames[i])
        cbf = np.expand_dims(plt.imread(normalcbfPath),0)
        cbv = np.expand_dims(plt.imread(normalcbvPath),0)
        dwi = np.expand_dims(plt.imread(normaldwiPath),0)
        t1c = np.expand_dims(plt.imread(normalt1cPath),0)
        t2 = np.expand_dims(plt.imread(normalt2Path),0)
        tmax = np.expand_dims(plt.imread(normaltmaxPath),0)
        ttp = np.expand_dims(plt.imread(normalttpPath),0)
        mask = np.expand_dims(plt.imread(normalmaskPath),0)
        for a, b, c, d, e, f, g, h, val in zip(
                            image_datagen.flow(cbf,batch_size=b_size,seed=seed,save_to_dir=save_here_cbf,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(cbv,batch_size=b_size,seed=seed,save_to_dir=save_here_cbv,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(dwi,batch_size=b_size,seed=seed,save_to_dir=save_here_dwi,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(t1c,batch_size=b_size,seed=seed,save_to_dir=save_here_t1c,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(t2,batch_size=b_size,seed=seed,save_to_dir=save_here_t2,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(tmax,batch_size=b_size,seed=seed,save_to_dir=save_here_tmax,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(ttp,batch_size=b_size,seed=seed,save_to_dir=save_here_ttp,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            mask_datagen.flow(mask,batch_size=b_size,seed=seed,save_to_dir=save_here_mask,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            range(ifi)) :
            #yield(x,y)
            k+=1

def validset(b_size):
    print('creating augmented validation images...')
    seed = 1337
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    save_here_cbf = 'data/augmented_data/validation/cbf'
    save_here_cbv = 'data/augmented_data/validation/cbv'
    save_here_dwi = 'data/augmented_data/validation/dwi'
    save_here_t1c = 'data/augmented_data/validation/t1c'
    save_here_t2 = 'data/augmented_data/validation/t2'
    save_here_tmax = 'data/augmented_data/validation/tmax'
    save_here_ttp = 'data/augmented_data/validation/ttp'
    save_here_mask = 'data/augmented_data/validation/mask'

    k=0
    for i in range(len(validImageNames)):
        normalcbfPath = 'data/normal_data/training/cbf/{}'.format(validImageNames[i])
        normalcbvPath = 'data/normal_data/training/cbv/{}'.format(validImageNames[i])
        normaldwiPath = 'data/normal_data/training/dwi/{}'.format(validImageNames[i])
        normalt1cPath = 'data/normal_data/training/t1c/{}'.format(validImageNames[i])
        normalt2Path = 'data/normal_data/training/t2/{}'.format(validImageNames[i])
        normaltmaxPath = 'data/normal_data/training/tmax/{}'.format(validImageNames[i])
        normalttpPath = 'data/normal_data/training/ttp/{}'.format(validImageNames[i])
        normalmaskPath = 'data/normal_data/training/mask/{}'.format(validImageNames[i])
        cbf = np.expand_dims(plt.imread(normalcbfPath),0)
        cbv = np.expand_dims(plt.imread(normalcbvPath),0)
        dwi = np.expand_dims(plt.imread(normaldwiPath),0)
        t1c = np.expand_dims(plt.imread(normalt1cPath),0)
        t2 = np.expand_dims(plt.imread(normalt2Path),0)
        tmax = np.expand_dims(plt.imread(normaltmaxPath),0)
        ttp = np.expand_dims(plt.imread(normalttpPath),0)
        mask = np.expand_dims(plt.imread(normalmaskPath),0)
        for a, b, c, d, e, f, g, h, val in zip(
                            image_datagen.flow(cbf,batch_size=b_size,seed=seed,save_to_dir=save_here_cbf,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(cbv,batch_size=b_size,seed=seed,save_to_dir=save_here_cbv,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(dwi,batch_size=b_size,seed=seed,save_to_dir=save_here_dwi,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(t1c,batch_size=b_size,seed=seed,save_to_dir=save_here_t1c,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(t2,batch_size=b_size,seed=seed,save_to_dir=save_here_t2,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(tmax,batch_size=b_size,seed=seed,save_to_dir=save_here_tmax,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            image_datagen.flow(ttp,batch_size=b_size,seed=seed,save_to_dir=save_here_ttp,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            mask_datagen.flow(mask,batch_size=b_size,seed=seed,save_to_dir=save_here_mask,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            range(ifi)) :
            #yield(x,y)
            k+=1


trainset(3)
validset(3)