import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import kmc2
from tqdm import tqdm

path = './BoWdata'



np_load_old = np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#np.load = np_load_old

all_train_des = np.load(path+'/save/all_train_des_3.npy')

print(all_train_des.shape)
des_list = []
for x in tqdm(range(len(all_train_des))):
    for des in all_train_des[x][0] :
        des_list.append(des)


print(np.array(des_list).shape)

codebook_len = 650

seeding = kmc2.kmc2(des_list,codebook_len)
clustered = MiniBatchKMeans(codebook_len,init=seeding,init_size=int(codebook_len*3.3)).fit(des_list)
codebook = clustered.cluster_centers_

np.save(path+'/save/codebook_%d_2.npy'%(codebook_len),codebook)
