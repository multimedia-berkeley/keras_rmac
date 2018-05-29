import cPickle as pickle


with open('rmac_sample.pickle', 'rb') as f:
    l = pickle.load(f)

print len(l)
print l[0]
