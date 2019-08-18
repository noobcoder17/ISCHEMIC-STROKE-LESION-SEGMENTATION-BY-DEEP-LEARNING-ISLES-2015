#normal training data extraction from .nii files
#leavinng all the black slides and corping the images
#only non black ot slides are considered
#each image dimension 192 x 192 and saved as .jpg format in grayscale


import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random2

data_dict = {
    "Flair" : "flair",
    "T1" : "t1",
    "T2" : "t2",
    "DWI" : "dwi",
    #"OT" : 'ot'
}
#data_types = ["Flair"]
data_types = ["OT"]

dir_path = "D:\\Machine Learning\\data\\SISS_2015\\data\\training"

folders = []
for i in range(1,29):
    folders.append(str(i))

prefix = ["%.2d" % i for i in range(1,100)]
imgNum = ["%.3d" % i for i in range(0,999999)]

upper_black_slide = 0
lower_black_slide = 0
new_subject = True
blank_slide_list = []

print("Saving Flair Images....")
img_name = []

c=-1
for folder in tqdm(folders):
    c+=1
    flairImg=0
    validIndex = []
    new_subject_list = []
    upper_black_slide = 0
    lower_black_slide = 0
    new_subject = True
    sub_folders = os.listdir(os.path.join(dir_path,folder))
    for sub_folder in sub_folders:
        file_names = os.listdir(os.path.join(dir_path,folder,sub_folder))
        for each_file in file_names:
            if ".nii" in each_file:
                data = nib.load(os.path.join(dir_path,folder,sub_folder,each_file))
                data = data.get_fdata().T
                if data_types[0] in each_file:
                    for i in range(data.shape[0]):
                        temp = np.sum(data[i])
                        if temp!=0:
                            new_subject = False
                            #name = str("./data/normal_data/training/mask/"+prefix[c]+"_"+imgNum[flairImg]+".jpg")
                            #name = str("./full_data/normal_data/training/mask/"+prefix[c]+"_"+imgNum[flairImg]+".jpg")
                            name = str("./full_data/normal_data/training/flair/"+prefix[c]+"_"+imgNum[flairImg]+".jpg")
                            ss = [str(prefix[c]+"_"+imgNum[flairImg]+".jpg")]
                            img_name.append(ss)
                            flairImg+=1
                            img = data[i]
                            img = img[19:211,19:211]
                            plt.imsave(name,img,cmap='gray')
                        else :
                            if new_subject :
                                upper_black_slide+=1
                            else:
                                lower_black_slide+=1
    new_subject_list.append(upper_black_slide)
    new_subject_list.append(lower_black_slide)
    blank_slide_list.append(new_subject_list)

np.save("./full_data/normal_data/training/imageNames",img_name)
#other data extract

for key in data_dict:
    print("Saving {} Images....".format(key))
    c=-1
    count = 0
    raw_data = []
    p = 0
    for folder in tqdm(folders):
        c+=1
        p+=1
        count=0
        sub_folders = os.listdir(os.path.join(dir_path,folder))
        for sub_folder in sub_folders:
            file_names = os.listdir(os.path.join(dir_path,folder,sub_folder))
            for each_file in file_names:
                if ".nii" in each_file:
                    data = nib.load(os.path.join(dir_path,folder,sub_folder,each_file))
                    data = data.get_fdata().T
                    if key in each_file:
                        for i in range(blank_slide_list[p-1][0],data.shape[0]-blank_slide_list[p-1][1]):
                            #name = str("./data/normal_data/training/"+str(key)+"/"+prefix[c]+"_"+imgNum[count]+".jpg")
                            name = str("./full_data/normal_data/training/"+str(key)+"/"+prefix[c]+"_"+imgNum[count]+".jpg")
                            img = data[i]
                            img = img[19:211,19:211]
                            plt.imsave(name,img,cmap='gray')                      
                            count+=1











