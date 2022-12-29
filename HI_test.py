from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np
from tqdm import tqdm

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_maxabs = MaxAbsScaler()
scaler_robust = RobustScaler()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


def HistogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m,n))
    for i in tqdm(range(m)):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result

test_histogram1 = np.load('./BoWdata/save/histogram_test_SPM1.npy')
test_histogram2 = np.load('./BoWdata/save/histogram_test_SPM2.npy')
test_histogram3 = np.load('./BoWdata/save/histogram_test_SPM3.npy')
test_histogram4 = np.load('./BoWdata/save/histogram_test_SPM4.npy')
test_histogram5 = np.load('./BoWdata/save/histogram_test_SPM5.npy')
test_histogram6 = np.load('./BoWdata/save/histogram_test_SPM6.npy')
test_histogram7 = np.load('./BoWdata/save/histogram_test_SPM7.npy')

histogram1 = np.load('./BoWdata/save/histogram_SPM1_2.npy')
histogram2 = np.load('./BoWdata/save/histogram_SPM2_2.npy')
histogram3 = np.load('./BoWdata/save/histogram_SPM3_2.npy')
histogram4 = np.load('./BoWdata/save/histogram_SPM4_2.npy')
histogram5 = np.load('./BoWdata/save/histogram_SPM5_2.npy')
histogram6 = np.load('./BoWdata/save/histogram_SPM6_2.npy')
histogram7 = np.load('./BoWdata/save/histogram_SPM7_2.npy')

scaler_standard.fit(histogram1)
histogram1 = scaler_standard.transform(histogram1)
train_HI1 = HistogramIntersection(test_histogram1,histogram1)
np.save('./BoWdata/save/test_HI1.npy',train_HI1)

scaler_standard.fit(histogram2)
histogram2 = scaler_standard.transform(histogram2)
train_HI2 = HistogramIntersection(test_histogram2,histogram2)
np.save('./BoWdata/save/test_HI2.npy',train_HI2)

scaler_standard.fit(histogram3)
histogram3 = scaler_standard.transform(histogram3)
train_HI3 = HistogramIntersection(test_histogram3,histogram3)
np.save('./BoWdata/save/test_HI3.npy',train_HI3)

scaler_standard.fit(histogram4)
histogram4 = scaler_standard.transform(histogram4)
train_HI4 = HistogramIntersection(test_histogram4,histogram4)
np.save('./BoWdata/save/test_HI4.npy',train_HI4)

scaler_standard.fit(histogram5)
histogram5 = scaler_standard.transform(histogram5)
train_HI5 = HistogramIntersection(test_histogram5,histogram5)
np.save('./BoWdata/save/test_HI5.npy',train_HI5)

scaler_standard.fit(histogram6)
histogram6 = scaler_standard.transform(histogram6)
train_HI6 = HistogramIntersection(test_histogram6,histogram6)
np.save('./BoWdata/save/test_HI6.npy',train_HI6)

scaler_standard.fit(histogram7)
histogram7 = scaler_standard.transform(histogram7)
train_HI7 = HistogramIntersection(test_histogram7,histogram7)
np.save('./BoWdata/save/test_HI7.npy',train_HI7)
