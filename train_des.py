
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook, tqdm
path = './BoWdata'
img_size = 288
jump = 32
fast = cv2.FastFeatureDetector_create()
sift = cv2.xfeatures2d.SIFT_create()

## 라벨링 정보를 불러와 사전형으로 저장
label_pd = pd.read_csv(path + '/Label2Names.csv',header = None)
label_data = np.array(label_pd)

label_dict = {}
for x in range(len(label_data)):
    label_dict[label_data[x][1]] = label_data[x][0]
label_dict['BACKGROUND_Google'] = 102

path_train = path + '/train/'
all_train_des = []
all_train_label = []

a = 0

cut0 = int((img_size)*(1/2))
cut1 = int((img_size)*(1/3))
cut2 = int((img_size)*(2/3))
cut3 = int((img_size)*(1/4))
cut4 = int((img_size)*(2/4))
cut5 = int((img_size)*(3/4))



print("zz")
for label in tqdm(os.listdir(path_train)):
    print(path_train+label)
    img = [None]*30
    for file in tqdm(os.listdir(path_train+label)):
        this_des = []
        img0 = cv2.imread(path_train+label+'/'+file,cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0,(img_size,img_size))
    
        #level 0
        img[0] = img0
        #level 1
        img[1] = img0[:cut0,:cut0]
        img[2] = img0[cut0:,:cut0]
        img[3] = img0[:cut0,cut0:]
        img[4] = img0[cut0:,cut0:]
        #level 2
        img[5] = img0[:cut1,:cut1]
        img[6] = img0[cut1:cut2,:cut1]
        img[7] = img0[cut2:,:cut1]
        img[8] = img0[:cut1,cut1:cut2]
        img[9] = img0[cut1:cut2,cut1:cut2]
        img[10] = img0[cut2:,cut1:cut2]
        img[11] = img0[:cut1,cut2:]
        img[12] = img0[cut1:cut2,cut2:]
        img[13] = img0[cut2:,cut2:]
        #level 3
        img[14] = img0[:cut3,:cut3]
        img[15] = img0[cut3:cut4,:cut3]
        img[16] = img0[cut4:cut5,:cut3]
        img[17] = img0[cut5:,:cut3]
        img[18] = img0[:cut3,cut3:cut4]
        img[19] = img0[cut3:cut4,cut3:cut4]
        img[20] = img0[cut4:cut5,cut3:cut4]
        img[21] = img0[cut5:,cut3:cut4]
        img[22] = img0[:cut3,cut4:cut5]
        img[23] = img0[cut3:cut4,cut4:cut5]
        img[24] = img0[cut4:cut5,cut4:cut5]
        img[25] = img0[cut5:,cut4:cut5]
        img[26] = img0[:cut3,cut5:]
        img[27] = img0[cut3:cut4,cut5:]
        img[28] = img0[cut4:cut5,cut5:]
        img[29] = img0[cut5:,cut5:]

        a = 0
        for x in img :
            kp = fast.detect(x)
            if len(kp) == 0:
                kp = [cv2.KeyPoint(x,y,jump) for x in range(0,img_size,jump) for y in range(0,img_size,jump)]
            _,des = sift.compute(x,kp)
                        
            if des is None:
                print(kp)
                print(len(kp))
                plt.imshow(x)
                exit()
            
            this_des.append(des)            
            a+=1

        all_train_des.append(this_des)   
        all_train_label.append(label_dict[label])
   
np.save(path+'/save/all_train_des_3.npy',all_train_des)
np.save(path+'/save/all_train_label.npy',all_train_label)
print('done')