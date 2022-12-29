import numpy as np
from tqdm import tqdm

from sklearn import svm


path = './BoWdata'
path_save = './BoWdata/save_3/'
codebook_len = 800
img_size = 288
jump = 6


all_train_label = np.load('./BoWdata/save_2/all_label.npy')



train_label = []
test_label = []
for x in range(102):
	for y in range(30):
		if y < 25 :
			train_label.append(all_train_label[x*30+y])
		else :
			test_label.append(all_train_label[x*30+y])





def check_score(predict,answer):
    score = 0
    for x in range(len(predict)):
        if (predict[x] == answer[x]):
            score = score + 1  
    print('%f%%'%((score/len(predict))*100))





x = 4

train_HI = np.load(path_save+'train_HI%d_%dx%d_%d.npy'%(x, img_size, img_size, jump))
test_HI = np.load(path_save+'test_HI%d_%dx%d_%d.npy'%(x, img_size, img_size, jump))

model = svm.SVC(kernel='precomputed').fit(train_HI, train_label)
predict = model.predict(test_HI)
check_score(predict, test_label)


