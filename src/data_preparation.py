import numpy as np
import h5py
import pickle
import theano
import theano.tensor as T


def preprocess_gabor_feature_maps(feat_dict, act_func=None, dtype=np.float32):
    '''
    Apply optional nonlinearity to the feature maps itself and concatenate feature maps of the same dimensions.
    Returns the feature maps and a list of theano variables to represent them, and the shape of the fmaps.
    '''
    fmap_rez = []
    for k in feat_dict.keys():
        fmap_rez += [feat_dict[k].shape[2],]
    resolutions = np.unique(fmap_rez)
    # concatenate and sort as list
    fmaps_res_count = len(resolutions)
    fmaps_count = 0
    fmaps, _fmaps = [], []
    for r in range(fmaps_res_count):
        fmaps  += [[],]
        _fmaps += [T.tensor4(),]     # theano symbols
    nonlinearity = act_func
    if nonlinearity is None:
        nonlinearity = lambda x: x
    for k in feat_dict.keys():
        # determine which resolution idx this map belongs to
        ridx = np.argmax(resolutions==feat_dict[k].shape[2])
        if len(fmaps[ridx])==0:
            fmaps[ridx] = nonlinearity(feat_dict[k].astype(dtype))
        else:
            fmaps[ridx] = np.concatenate((fmaps[ridx], nonlinearity(feat_dict[k].astype(dtype))), axis=1)       
        fmaps_count += 1
    fmaps_sizes = [] 
    for fmap in fmaps:
        fmaps_sizes += [fmap.shape]
    print fmaps_sizes
    print "total fmaps = %d" % fmaps_count 
    return fmaps, _fmaps, fmaps_sizes 



def save_stuff(save_to_this_file, data_objects_dict):
    failed = []
    with h5py.File(save_to_this_file+'.h5py', 'w') as hf:
        for k,v in data_objects_dict.iteritems():
            try:
                hf.create_dataset(k,data=v)
                print 'saved %s in h5py file' %(k)
            except:
                failed.append(k)
                print 'failed to save %s as h5py. will try pickle' %(k)   
    for k in failed:
        with open(save_to_this_file+'_'+'%s.pkl' %(k), 'w') as pkl:
            try:
                pickle.dump(data_objects_dict[k],pkl)
                print 'saved %s as pkl' %(k)
            except:
                print 'failed to save %s in any format. lost.' %(k) 
