import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import kmc2
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn import svm

sift = cv2.xfeatures2d.SIFT_create()
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_maxabs = MaxAbsScaler()
scaler_robust = RobustScaler()

path = './BoWdata'
path_save = './BoWdata/save_2/'
codebook_len = 800
img_size = 288
jump = 6

'''
label_pd = pd.read_csv(path + '/Label2Names.csv',header = None)
label_data = np.array(label_pd)

label_dict = {}
for x in range(len(label_data)):
	label_dict[label_data[x][1]] = label_data[x][0]
label_dict['BACKGROUND_Google'] = 102

path_train = path+'/train/'
all_train_des = []
all_train_label = []
a = 0
for label in os.listdir(path_train):
    print(path_train+label,label_dict[label])
    for file in os.listdir(path_train+label):
        img = cv2.imread(path_train+label+'/'+file,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(img_size,img_size))    
        kp = [cv2.KeyPoint(x,y,jump) for x in range(0,img.shape[1],jump) for y in range(0,img.shape[0],jump)]    
        _,des =sift.compute(img,kp)
        all_train_des.append(des)
        all_train_label.append(label_dict[label])
    a+=1
    print(int((a/len(os.listdir(path_train)))*100+0.5),'%')

np.save(path_save+'/all_des_%dx%d_%d.npy'%(img_size,img_size,jump),all_train_des)
np.save(path_save+'/all_label.npy',all_train_label)
'''
all_train_des = np.load(path_save+'/all_des_%dx%d_%d.npy'%(img_size,img_size,jump))
all_train_label = np.load(path_save+'/all_label.npy')

# cut background
train_des = []
train_label = []
for x in range(len(all_train_des)):
    if(all_train_label[x]!=102):
        train_des.append(all_train_des[x])
        train_label.append(all_train_label[x])
    
print(np.array(train_des).shape,np.array(train_label).shape)

des_list= []
for x in range(len(all_train_des)) :
  des_list.extend(all_train_des[x])
'''
#make codebook
seeding = kmc2.kmc2(des_list,codebook_len)
clustered_mini = MiniBatchKMeans(codebook_len,init=seeding, init_size=int(codebook_len*3.3)).fit(des_list)
codebook = clustered_mini.cluster_centers_

np.save(path_save+'/all_codebook_%dx%d_%d_%d.npy'%(img_size, img_size, jump, codebook_len),codebook)
'''
codebook = np.load(path+'/all_codebook_%dx%d_%d_%d.npy'%(img_size, img_size, jump, codebook_len))

## 각각의 이미지의 코드북에대한 히스토그램
def SPM2(des, codebook, codebook_len, img_size, step) :
    
	cut1 = int((img_size / step)*(1 / 4))
	cut2 = int((img_size / step)*(2 / 4))
	cut3 = int((img_size / step)*(3 / 4))
	cut4 = int((img_size / step)*(1 / 3))
	cut5 = int((img_size / step)*(2 / 3))

	histogram1=[]
	histogram2=[]
	histogram3=[]
	histogram4=[]
	histogram5=[]

	a = 0
	for x in tqdm(range(len(des))) :
		des_map = des[x].reshape(int(img_size / step), int(img_size / step), 128)

        ## level 0
		lv_0, _ = vq(des[x], codebook)
        #
		lv_0, _ = np.histogram(lv_0, bins = list(range(codebook_len + 1)))

        #
        ## level 1
		lv_1_1, _ = vq(des_map[:cut2, : cut2, : ].reshape(-1, 128), codebook)
		lv_1_1, _ = np.histogram(lv_1_1, bins = list(range(codebook_len + 1)))
		lv_1_1 = lv_1_1.reshape((codebook_len, 1))

		lv_1_2, _ = vq(des_map[cut2:, : cut2, : ].reshape(-1, 128), codebook)
		lv_1_2, _ = np.histogram(lv_1_2, bins = list(range(codebook_len + 1)))
		lv_1_2 = lv_1_2.reshape((codebook_len, 1))

		lv_1_3, _ = vq(des_map[:cut2, cut2 : , : ].reshape(-1, 128), codebook)
		lv_1_3, _ = np.histogram(lv_1_3, bins = list(range(codebook_len + 1)))
		lv_1_3 = lv_1_3.reshape((codebook_len, 1))

		lv_1_4, _ = vq(des_map[cut2:, cut2 : , : ].reshape(-1, 128), codebook)
		lv_1_4, _ = np.histogram(lv_1_4, bins = list(range(codebook_len + 1)))
		lv_1_4 = lv_1_4.reshape((codebook_len, 1))
        #

		lv_1 = np.concatenate((lv_1_1, lv_1_2, lv_1_3, lv_1_4), axis = 1).flatten()

		#level 1.5
		lv_3_1, _ = vq(des_map[:cut4, : cut4, : ].reshape(-1, 128), codebook)
		lv_3_1, _ = np.histogram(lv_3_1, bins = list(range(codebook_len + 1)))
		lv_3_1 = lv_3_1.reshape((codebook_len, 1))

		lv_3_2, _ = vq(des_map[cut4:cut5, : cut4, : ].reshape(-1, 128), codebook)
		lv_3_2, _ = np.histogram(lv_3_2, bins = list(range(codebook_len + 1)))
		lv_3_2 = lv_3_2.reshape((codebook_len, 1))

		lv_3_3, _ = vq(des_map[cut5:, : cut4, : ].reshape(-1, 128), codebook)
		lv_3_3, _ = np.histogram(lv_3_3, bins = list(range(codebook_len + 1)))
		lv_3_3 = lv_3_3.reshape((codebook_len, 1))

		lv_3_4, _ = vq(des_map[:cut4, cut4 : cut5, : ].reshape(-1, 128), codebook)
		lv_3_4, _ = np.histogram(lv_3_4, bins = list(range(codebook_len + 1)))
		lv_3_4 = lv_3_4.reshape((codebook_len, 1))

		lv_3_5, _ = vq(des_map[cut4:cut5, cut4 : cut5, : ].reshape(-1, 128), codebook)
		lv_3_5, _ = np.histogram(lv_3_5, bins = list(range(codebook_len + 1)))
		lv_3_5 = lv_3_5.reshape((codebook_len, 1))

		lv_3_6, _ = vq(des_map[cut5:, cut4 : cut5, : ].reshape(-1, 128), codebook)
		lv_3_6, _ = np.histogram(lv_3_6, bins = list(range(codebook_len + 1)))
		lv_3_6 = lv_3_6.reshape((codebook_len, 1))

		lv_3_7, _ = vq(des_map[:cut4, cut5 : , : ].reshape(-1, 128), codebook)
		lv_3_7, _ = np.histogram(lv_3_7, bins = list(range(codebook_len + 1)))
		lv_3_7 = lv_3_7.reshape((codebook_len, 1))

		lv_3_8, _ = vq(des_map[cut4:cut5, cut5 : , : ].reshape(-1, 128), codebook)
		lv_3_8, _ = np.histogram(lv_3_8, bins = list(range(codebook_len + 1)))
		lv_3_8 = lv_3_8.reshape((codebook_len, 1))

		lv_3_9, _ = vq(des_map[cut5:, cut5 : , : ].reshape(-1, 128), codebook)
		lv_3_9, _ = np.histogram(lv_3_9, bins = list(range(codebook_len + 1)))
		lv_3_9 = lv_3_9.reshape((codebook_len, 1))

		lv_3 = np.concatenate((lv_3_1, lv_3_2, lv_3_3, lv_3_4, lv_3_5, lv_3_6, lv_3_7, lv_3_8, lv_3_9), axis = 1).flatten()


        ## level 2
		lv_2_1, _ = vq(des_map[:cut1, : cut1, : ].reshape(-1, 128), codebook)
		lv_2_1, _ = np.histogram(lv_2_1, bins = list(range(codebook_len + 1)))
		lv_2_1 = lv_2_1.reshape((codebook_len, 1))
		lv_2_2, _ = vq(des_map[cut1:, : cut1, : ].reshape(-1, 128), codebook)
		lv_2_2, _ = np.histogram(lv_2_2, bins = list(range(codebook_len + 1)))
		lv_2_2 = lv_2_2.reshape((codebook_len, 1))
		lv_2_3, _ = vq(des_map[cut2:cut3, : cut1, : ].reshape(-1, 128), codebook)
		lv_2_3, _ = np.histogram(lv_2_3, bins = list(range(codebook_len + 1)))
		lv_2_3 = lv_2_3.reshape((codebook_len, 1))
		lv_2_4, _ = vq(des_map[cut3:, : cut1, : ].reshape(-1, 128), codebook)
		lv_2_4, _ = np.histogram(lv_2_4, bins = list(range(codebook_len + 1)))
		lv_2_4 = lv_2_4.reshape((codebook_len, 1))
        #
		lv_2_5, _ = vq(des_map[:cut1, cut1 : cut2, : ].reshape(-1, 128), codebook)
		lv_2_5, _ = np.histogram(lv_2_5, bins = list(range(codebook_len + 1)))
		lv_2_5 = lv_2_5.reshape((codebook_len, 1))
		lv_2_6, _ = vq(des_map[cut1:, cut1 : cut2, : ].reshape(-1, 128), codebook)
		lv_2_6, _ = np.histogram(lv_2_6, bins = list(range(codebook_len + 1)))
		lv_2_6 = lv_2_6.reshape((codebook_len, 1))
		lv_2_7, _ = vq(des_map[cut2:cut3, cut1 : cut2, : ].reshape(-1, 128), codebook)
		lv_2_7, _ = np.histogram(lv_2_7, bins = list(range(codebook_len + 1)))
		lv_2_7 = lv_2_7.reshape((codebook_len, 1))
		lv_2_8, _ = vq(des_map[cut3:, cut1 : cut2, : ].reshape(-1, 128), codebook)
		lv_2_8, _ = np.histogram(lv_2_8, bins = list(range(codebook_len + 1)))
		lv_2_8 = lv_2_8.reshape((codebook_len, 1))
        #
		lv_2_9, _ = vq(des_map[:cut1, cut2 : cut3, : ].reshape(-1, 128), codebook)
		lv_2_9, _ = np.histogram(lv_2_9, bins = list(range(codebook_len + 1)))
		lv_2_9 = lv_2_9.reshape((codebook_len, 1))
		lv_2_10, _ = vq(des_map[cut1:, cut2 : cut3, : ].reshape(-1, 128), codebook)
		lv_2_10, _ = np.histogram(lv_2_10, bins = list(range(codebook_len + 1)))
		lv_2_10 = lv_2_10.reshape((codebook_len, 1))
		lv_2_11, _ = vq(des_map[cut2:cut3, cut2 : cut3, : ].reshape(-1, 128), codebook)
		lv_2_11, _ = np.histogram(lv_2_11, bins = list(range(codebook_len + 1)))
		lv_2_11 = lv_2_11.reshape((codebook_len, 1))
		lv_2_12, _ = vq(des_map[cut3:, cut2 : cut3, : ].reshape(-1, 128), codebook)
		lv_2_12, _ = np.histogram(lv_2_12, bins = list(range(codebook_len + 1)))
		lv_2_12 = lv_2_12.reshape((codebook_len, 1))
        #
		lv_2_13, _ = vq(des_map[:cut1, cut3 : , : ].reshape(-1, 128), codebook)
		lv_2_13, _ = np.histogram(lv_2_13, bins = list(range(codebook_len + 1)))
		lv_2_13 = lv_2_13.reshape((codebook_len, 1))
		lv_2_14, _ = vq(des_map[cut1:, cut3 : , : ].reshape(-1, 128), codebook)
		lv_2_14, _ = np.histogram(lv_2_14, bins = list(range(codebook_len + 1)))
		lv_2_14 = lv_2_14.reshape((codebook_len, 1))
		lv_2_15, _ = vq(des_map[cut2:cut3, cut3 : , : ].reshape(-1, 128), codebook)
		lv_2_15, _ = np.histogram(lv_2_15, bins = list(range(codebook_len + 1)))
		lv_2_15 = lv_2_15.reshape((codebook_len, 1))
		lv_2_16, _ = vq(des_map[cut3:, cut3 : , : ].reshape(-1, 128), codebook)
		lv_2_16, _ = np.histogram(lv_2_16, bins = list(range(codebook_len + 1)))
		lv_2_16 = lv_2_16.reshape((codebook_len, 1))
        #

		lv_2 = np.concatenate((lv_2_1, lv_2_2, lv_2_3, lv_2_4, lv_2_5, lv_2_6, lv_2_7, lv_2_8, lv_2_9, lv_2_10, lv_2_11, lv_2_12, lv_2_13, lv_2_14, lv_2_15, lv_2_16), axis = 1).flatten()
		


		hist = np.concatenate((lv_0*(0.2), lv_1*(0.3), lv_3*(0.4), lv_2*(0.5)))
		histogram1.append(hist)
		hist = np.concatenate((lv_0*(0.2), lv_1*(0.4), lv_3*(0.6), lv_2*(0.8)))
		histogram2.append(hist)
		hist = np.concatenate((lv_0*(0.25), lv_1*(0.25), lv_3*(0.25), lv_2*(0.25)))
		histogram3.append(hist)
		hist = np.concatenate((lv_0*(0.15), lv_1*(0.3), lv_3*(0.45), lv_2*(0.6)))
		histogram4.append(hist)
		hist = np.concatenate((lv_0*(0.25), lv_1*(0.25), lv_3*(0.5), lv_2*(0.5)))
		histogram5.append(hist)
		

	return (histogram2)


def HistogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m,n))
    for i in tqdm(range(m)):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result



print('Build SPM')

#train_histogram = SPM2(train_des,codebook,codebook_len,img_size,jump)

#np.save(path_save+'all_train_histogram_%dx%d_%d_%d.npy'%(img_size, img_size,jump,codebook_len),train_histogram)


train_histogram = np.load(path_save+'all_train_histogram_%dx%d_%d_%d.npy'%(img_size, img_size,jump,codebook_len))


print('Build HI')
scaler_standard.fit(train_histogram)
train_histogram_scaled = scaler_standard.transform(train_histogram)
train_HI = HistogramIntersection(train_histogram_scaled, train_histogram_scaled)

np.save(path_save+'nb_train_HI_%dx%d_%d_%d.npy'%(img_size, img_size,jump,codebook_len),train_HI)
print('Learn')
model = svm.SVC(kernel='precomputed').fit(train_HI, train_label)



# 테스트 이미지를 불러오며 각 이미지의 이름과 기술정보를 저장
path_test = path + '/test/'
test_des = []
test_file_name = []
a = 0
for file in tqdm(os.listdir(path_test)):
	if(file[0]=='i'):
  		img = cv2.imread(path_test+file,cv2.IMREAD_GRAYSCALE)
  		img = cv2.resize(img,(img_size,img_size))
  		kp = [cv2.KeyPoint(x,y,jump) for x in range(0,img.shape[1],jump) for y in range(0,img.shape[0],jump)] 
  		_,des =sift.compute(img,kp)
  		test_des.append(des)
  		test_file_name.append(file)
  

print('test SPM')
test_histogram = SPM2(test_des,codebook,codebook_len,img_size,jump)
scaler_standard.fit(test_histogram)
test_histogram_scaled = scaler_standard.transform(test_histogram)
print('test HI')
test_HI = HistogramIntersection(test_histogram_scaled, train_histogram_scaled)


predict = model.predict(test_HI)
    


submit_form = np.array(pd.read_csv('./BoWdata/submit.csv'))
test_file_name = np.load('./BoWdata/save/test_file_name.npy')

def save(predict,form,file_list,filename):
    

    submit_dict = {}
    for x in range(len(form)):
        submit_dict[form[x][0]] = form[x][1]

    for x in range (len(file_list)):
        submit_dict[file_list[x]] = predict[x]

    submit_zip = zip(submit_dict.keys(), submit_dict.values())
    submit_list = list(submit_zip)

    save_file = pd.DataFrame(submit_list)
    save_file.to_csv(('./submission_'+filename+'.csv'),header=['Id','Category'],index = False)
    print('save is done')

save(predict,submit_form,test_file_name,'650')