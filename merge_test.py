import cPickle as pickle

with open('rmac_test_backup.pickle', 'rb') as f:
    result_1 = pickle.load(f)

with open('rmac_test_backup2.pickle', 'rb') as f:
    result_2 = pickle.load(f)

filelist = result_1[0] + result_2[0]
rmac = result_1[1] + result_2[1]
with open('rmac_test_merged.pickle', 'wb') as f:
    pickle.dump((filelist, rmac), f)

print len(filelist), len(rmac)
