from sklearn import svm
import numpy as np
import pandas as pd
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_label = np.load('./BoWdata/save/all_train_label.npy')

train_HI1 = np.load('./BoWdata/save/train_HI1.npy')
train_HI2 = np.load('./BoWdata/save/train_HI2.npy')
train_HI3 = np.load('./BoWdata/save/train_HI3.npy')
train_HI4 = np.load('./BoWdata/save/train_HI4.npy')
train_HI5 = np.load('./BoWdata/save/train_HI5.npy')
train_HI6 = np.load('./BoWdata/save/train_HI6.npy')
train_HI7 = np.load('./BoWdata/save/train_HI7.npy')

model1 = svm.SVC(kernel='precomputed').fit(train_HI1, train_label)
model2 = svm.SVC(kernel='precomputed').fit(train_HI2, train_label)
model3 = svm.SVC(kernel='precomputed').fit(train_HI3, train_label)
model4 = svm.SVC(kernel='precomputed').fit(train_HI4, train_label)
model5 = svm.SVC(kernel='precomputed').fit(train_HI5, train_label)
model6 = svm.SVC(kernel='precomputed').fit(train_HI6, train_label)
model7 = svm.SVC(kernel='precomputed').fit(train_HI7, train_label)

test_HI1 = np.load('./BoWdata/save/test_HI1.npy')
test_HI2 = np.load('./BoWdata/save/test_HI2.npy')
test_HI3 = np.load('./BoWdata/save/test_HI3.npy')
test_HI4 = np.load('./BoWdata/save/test_HI4.npy')
test_HI5 = np.load('./BoWdata/save/test_HI5.npy')
test_HI6 = np.load('./BoWdata/save/test_HI6.npy')
test_HI7 = np.load('./BoWdata/save/test_HI7.npy')

predict1 = model1.predict(test_HI1)
predict2 = model2.predict(test_HI2)
predict3 = model3.predict(test_HI3)
predict4 = model4.predict(test_HI4)
predict5 = model5.predict(test_HI5)
predict6 = model6.predict(test_HI6)
predict7 = model7.predict(test_HI7)




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


def check_score(predict,answer):
    score = 0
    for x in range(len(predict)):
        if (predict[x] == answer[x]):
            score = score + 1  
    print('%.2f%%'%((score/len(predict))*100))

#check_score(predict1,train_label)
#check_score(predict2,train_label)
#check_score(predict3,train_label)
#check_score(predict4,train_label)



save(predict1,submit_form,test_file_name,'1')
save(predict2,submit_form,test_file_name,'2')
save(predict3,submit_form,test_file_name,'3')
save(predict4,submit_form,test_file_name,'4')
save(predict5,submit_form,test_file_name,'5')
save(predict6,submit_form,test_file_name,'6')
save(predict7,submit_form,test_file_name,'7')

