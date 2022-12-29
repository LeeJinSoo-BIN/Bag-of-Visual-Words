import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

a0 = np.load('./BoWdata/save_2/test_histogram0_288x288_6_800.npy')
a1 = np.load('./BoWdata/save_2/test_histogram1_288x288_6_800.npy')
a2 = np.load('./BoWdata/save_2/test_histogram2_288x288_6_800.npy')
a3 = np.load('./BoWdata/save_2/test_histogram3_288x288_6_800.npy')
a4 = np.load('./BoWdata/save_2/test_histogram4_288x288_6_800.npy')

print(a0-a4)


