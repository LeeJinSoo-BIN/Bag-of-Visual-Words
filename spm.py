from scipy.cluster.vq import vq
import numpy as np
from tqdm import tqdm
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


codebook_len = 650
codebook = np.load('./BoWdata/save/codebook_%d_2.npy'%(650))
print(codebook.shape)

all_train_des = np.load('./BoWdata/save/all_train_des_3.npy')
print(all_train_des.shape)
print(all_train_des[1][1])
print(all_train_des[1][2])
print(all_train_des[1][3])
print(all_train_des[1][4])
print(all_train_des[1][5])
print(all_train_des[1][1].shape)
print(all_train_des[1][2].shape)
print(all_train_des[1][3].shape)
print(all_train_des[1][4].shape)
print(all_train_des[1][5].shape)


histogram1 = []
histogram2 = []
histogram3 = []
histogram4 = []
histogram5 = []
histogram6 = []
histogram7 = []
for x in tqdm(range(len(all_train_des))):
    lv_1_ = [None]*30
    lv_2_ = [None]*30
    lv_3_ = [None]*30
    #level_0
    lv_0,_ = vq(all_train_des[x][0].reshape(-1,128), codebook)    
    lv_0,_ = np.histogram(lv_0, bins=list(range(codebook_len+1)))

    #level_1
    for a in range(1,5):    
        lv_1_[a],_ = vq(all_train_des[x][a].reshape(-1,128), codebook)    
        lv_1_[a],_ = np.histogram(lv_1_[a], bins=list(range(codebook_len+1)))
        lv_1_[a] = lv_1_[a].reshape((codebook_len,1))
    
    lv_1 = np.concatenate((lv_1_[1], lv_1_[2], lv_1_[3], lv_1_[4]), axis=1).flatten()
    #level_2
    for a in range(5,14):  
        lv_2_[a],_ = vq(all_train_des[x][a].reshape(-1,128), codebook)    
        lv_2_[a],_ = np.histogram(lv_2_[a], bins=list(range(codebook_len+1)))
        lv_2_[a] = lv_2_[a].reshape((codebook_len,1))
    lv_2 = np.concatenate((lv_2_[5], lv_2_[6],lv_2_[7], lv_2_[8],
                          lv_2_[9], lv_2_[10], lv_2_[11], lv_2_[12], lv_2_[13]), axis=1).flatten()
    #leve_3
    for a in range(14,30):
        lv_3_[a],_ = vq(all_train_des[x][a].reshape(-1,128), codebook)    
        lv_3_[a],_ = np.histogram(lv_3_[a], bins=list(range(codebook_len+1)))
        lv_3_[a] = lv_3_[a].reshape((codebook_len,1))
    lv_3 = np.concatenate((lv_3_[14], lv_3_[15], lv_3_[16], lv_3_[17], lv_3_[18], lv_3_[19], lv_3_[20],
                        lv_3_[21], lv_3_[22], lv_3_[23], lv_3_[24], lv_3_[25], lv_3_[26], lv_3_[27], lv_3_[28], lv_3_[29]), axis=1).flatten()

    hist = np.concatenate((lv_0*(0.2), lv_1*(0.3), lv_2*(0.4), lv_3*(0.5)))
    histogram1.append(hist)
    hist = np.concatenate((lv_0, lv_1, lv_2, lv_3))
    histogram2.append(hist)
    hist = np.concatenate((lv_0*(0.25), lv_1*(0.25), lv_2*(0.25), lv_3*(0.25)))
    histogram3.append(hist)
    hist = np.concatenate((lv_0*(0.2), lv_1*(0.4), lv_2*(0.6), lv_3*(0.8)))
    histogram4.append(hist)
    hist = np.concatenate((lv_0*(0.1), lv_1*(0.2), lv_2*(0.3), lv_3*(0.4)))
    histogram5.append(hist)
    hist = np.concatenate((lv_0*(0.15), lv_1*(0.3), lv_2*(0.45), lv_3*(0.6)))
    histogram6.append(hist)
    hist = np.concatenate((lv_0*(0.2), lv_1*(0.25), lv_2*(0.3), lv_3*(0.35)))
    histogram7.append(hist)
    

np.save('./BoWdata/save/histogram_SPM1_2.npy',histogram1)
np.save('./BoWdata/save/histogram_SPM2_2.npy',histogram2)
np.save('./BoWdata/save/histogram_SPM3_2.npy',histogram3)
np.save('./BoWdata/save/histogram_SPM4_2.npy',histogram4)
np.save('./BoWdata/save/histogram_SPM5_2.npy',histogram5)
np.save('./BoWdata/save/histogram_SPM6_2.npy',histogram6)
np.save('./BoWdata/save/histogram_SPM7_2.npy',histogram7)

