import cPickle as pickle

output_file = 'rmac_train_aj.pickle' 
INPUT_FILE = 'train_aj_group_by_size.pickle'
with open('/g/g92/choi13/src/keras_rmac/rmac_result/backup/'+ output_file, 'rb') as f:
    partial_result = pickle.load(f)
    completed_files = partial_result[0]
    rmac_result = partial_result[1]
    print len(completed_files), len(rmac_result)
    completed_files = set(completed_files)

with open('/g/g92/choi13/projects/landmark/' + INPUT_FILE, 'rb') as f:
    d = pickle.load(f)

filename_output = list()
len_by_key = [(key, len(d[key])) for key in d.keys()]
len_by_key = sorted(len_by_key, key=lambda x:x[1], reverse = True)
print len_by_key[:2]
for size, _ in len_by_key:
    filelist = d[size]
    for filename in filelist:
        if filename in completed_files:
            filename_output.append(filename)

print len(filename_output), len(rmac_result)
with open(output_file, 'wb') as f:
    pickle.dump((filename_output, rmac_result), f)
    
