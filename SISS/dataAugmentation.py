from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

ign = np.load('./data/normal_data/data/dataset_2/training/imageNames.npy')
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

    save_here_img = 'data/augmented_data/training/image'
    save_here_mask = 'data/augmented_data/training/mask'

    k=0
    for i in range(len(trainImageNames)):
        normalimgPath = 'data/normal_data/data/dataset_2/training/image/{}'.format(trainImageNames[i])
        normalmaskPath = 'data/normal_data/data/dataset_2/training/mask/{}'.format(trainImageNames[i])
        img = np.expand_dims(plt.imread(normalimgPath),0)
        mask = np.expand_dims(plt.imread(normalmaskPath),0)
        for x, y, val in zip(image_datagen.flow(img,batch_size=b_size,seed=seed,save_to_dir=save_here_img,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            mask_datagen.flow(mask,batch_size=b_size,seed=seed,save_to_dir=save_here_mask,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            range(ifi)) :
            #yield(x,y)
            k+=1

def validset(b_size):
    print('creating augmented validation images...')
    seed = 1243
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    save_here_img = 'data/augmented_data/validation/image'
    save_here_mask = 'data/augmented_data/validation/mask'

    k=0
    for i in range(len(validImageNames)):
        normalimgPath = 'data/normal_data/data/dataset_2/training/image/{}'.format(validImageNames[i])
        normalmaskPath = 'data/normal_data/data/dataset_2/training/mask/{}'.format(validImageNames[i])
        img = np.expand_dims(plt.imread(normalimgPath),0)
        mask = np.expand_dims(plt.imread(normalmaskPath),0)
        for x, y, val in zip(image_datagen.flow(img,batch_size=b_size,seed=seed,save_to_dir=save_here_img,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            mask_datagen.flow(mask,batch_size=b_size,seed=seed,save_to_dir=save_here_mask,save_prefix='aug_{}'.format(str(k)),save_format='jpg'),
                            range(ifi)) :
            #yield(x,y)
            k+=1


trainset(3)
validset(3)