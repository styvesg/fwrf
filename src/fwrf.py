########################################################################
### PACKAGE VERSIONS:												 ###
### theano: 	0.9  												 ###
### lasagne: 	0.2dev1												 ###
### numpy:
###                               
########################################################################

import sys
import struct
import time
import numpy as np
from tqdm import tqdm
import pickle
import math

import theano
import theano.tensor as T

import lasagne
import lasagne.layers as L
import lasagne.regularization as R
import lasagne.nonlinearities as NL
import lasagne.objectives as O
import lasagne.init as I

import numpy_utility as pnu
import lasagne_utility as plu


fpX = np.float32
print "theano floatX: %s" % theano.config.floatX
print "numpy floatX: %s" % fpX

########################################################################
###              SUPPORT FUNCTIONS                                   ###
########################################################################

class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=np.float32):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
    
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=np.float32, **kwargs):
        super(linspace, self).__init__(n_div, dtype=np.float32, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = width/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = width/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=np.float32, **kwargs):
        super(logspace, self).__init__(n_div, dtype=np.float32, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)



def iterate_range(start, length, batchsize):
    batch_count = length // batchsize 
    residual = length % batchsize
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual

def iterate_bounds(start, length, batchsize):
    batch_count = length // batchsize 
    residual = length % batchsize
    for i in range(batch_count):
        yield [start+i*batchsize, start+(i+1)*batchsize], batchsize
    if(residual>0):
        yield [start+batch_count*batchsize, start+length], residual	

def iterate_slice(start, length, batchsize):
    batch_count = length // batchsize 
    residual = length % batchsize
    for i in range(batch_count):
        yield slice(start+i*batchsize, start+(i+1)*batchsize), batchsize
    if(residual>0):
        yield slice(start+batch_count*batchsize,start+length), residual
        
def slice_arraylist(inputs, excerpt):            
    return [i[excerpt] for i in inputs]  

def iterate_minibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs), batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]
        
def iterate_multiminibatches(inputs, targets, batchsize):
    '''return inputs.shape[0]//batchsize batches plus one residual batches smaller than batchsize if needed'''
    length = len(targets)
    batch_count = len(targets) // batchsize 
    residual = length % batchsize    
    for start_idx in range(0, length-residual, batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield [i[excerpt] for i in inputs] + [targets[excerpt]]
    if(residual>0):
        excerpt = slice(length-residual, length)
        yield [i[excerpt] for i in inputs] + [targets[excerpt]]


def unique_rel_to_abs_models(rel_models, rx, ry, rs):
    '''converts a list of relative models to the absolute model specified by the range parameters rx, ry, rs
    returns a matrix of size (number of models, 3)
    '''
    nv = len(rel_models)
    nx, ny, ns = len(rx[1]), len(ry[1]), len(rs[1])
    assert nv==len(rx[0])
    ixs, iys, iss = np.unravel_index(rel_models, (nx, ny, ns))
    abs_models = np.ndarray(shape=(nv, 3), dtype=fpX)
    for v in range(nv):
        abs_models[v] = [rx[v,ixs[v]], ry[v,iys[v]], rs[v,iss[v]]]
    return abs_models

def rel_to_abs_shared_models(rel_models, rx, ry, rs):
    '''converts a list of relative models to the absolute model specified by the range parameters rx, ry, rs
    returns a matrix of size (number of models, 3)'''
    nv = len(rel_models)
    nx, ny, ns = len(rx), len(ry), len(rs)
    ixs, iys, iss = np.unravel_index(rel_models, (nx, ny, ns))
    return np.stack([rx[ixs[:]], ry[iys[:]], rs[iss[:]]], axis=1)


def pRF(fwrf_weights, fmap_rf, pool_rf):
    '''
    fwrf_weights is [nv, nf]
    fmap_rf is [nf] i.e. it specifies a gaussian sigma value for each feature map
    pool_rf is [nv, 3] i.e. it specifies a gaussian population pooling fct for each voxel 
    
    returns [nv,3], a rf x, y, and sigma for each voxel
    '''
    # we'd be better off performing the implicit convolution first.
    vsigma = np.zeros(shape=(nv), dtype=fpX)
    for v in pool_rf:
        vsigma[v] = np.average(np.sqrt(np.square(fmap_rf) + np.square(pool_rf[v,2,np.newaxis])), weights=fwrf_weights[v,:])
    return np.stack([pool_rf[:,0:1], pool_rf[:,1:2], vsigma[:,np.newaxis]], axis=1)

########################################################################
###                                                                  ###
########################################################################

def create_shared_batched_feature_maps_gaussian_weights(fmap_sizes, batch_v, batch_t, verbose=True):
    nf = 0
    _smsts = []
    mem_approx = 0
    rep_approx = 0
    for i,a in enumerate(fmap_sizes):
        nf += a[1]
        n_pix = a[2]
        assert n_pix==a[3], "Non square feature map not supported"
        _smsts += [theano.shared(np.zeros(shape=(batch_v, batch_t, n_pix, n_pix), dtype=fpX)),]
        mem_approx += 4*batch_v*batch_t*n_pix**2
        rep_approx += 4*a[1]*n_pix**2
        if verbose:
            print "> feature map candidates %d with shape %s" % (i, (batch_v, batch_t, n_pix, n_pix))
    if verbose:        
        print "  total number of feature maps = %d, in %d layers" % (nf, len(fmap_sizes))
        print "  feature map candidate using approx %.1f Mb of memory (VRAM and RAM)" % (fpX(mem_approx) /(1024*1024))
    return _smsts, nf

def set_shared_batched_feature_maps_gaussian_weights(_psmsts, xs, ys, ss, size=20.):
    '''
    The interpretation of receptive field weight factor is that they correspond, for each voxel, to the probability of this voxel of seeing 
    (through the weighted average) a given feature map pixel through its receptive field size and position in visual space. 
    Whether that feature map pixel is relevant to the representation of that particular voxel is left to the voxel encoding model to decide.
    '''
    nf = 0
    (nv, nt) = (len(xs), 1) if xs.ndim==1 else xs.shape[0:2]
    (sv, st) = _psmsts[0].get_value().shape[0:2]
    assert nv==sv and nt==st, "non conformal (%d,%d)!=(%d,%d)" % (nv, nt, sv, st)
    for a in _psmsts:
        n_pix = a.get_value().shape[2]
        _,_,mst = pnu.make_gaussian_mass_stack(xs.flatten(), ys.flatten(), ss.flatten(), n_pix, size=size, dtype=fpX)
        a.set_value(mst.reshape((nv, nt, n_pix, n_pix)))
    return _psmsts
    
def set_shared_parameters(shared_vars, values):
    for i,var in enumerate(shared_vars):
        var.set_value(values[i])    





########################################################################
###              SPECIAL LASAGNE LAYER AND MODEL                     ###
########################################################################
class pvFWRFLayer(L.Layer):
    '''
    pvFWRFLayer is a new lasagne layer for 'per voxel (pv)' candidate receptive field models. It assumes an input
    of shape (bn, nf, bv, bt) where bn is a batch of the time series, nf are the total number of features, bv is a batch of voxels and bt is a batch of candidate rf.

    The return values correspond to the predicted voxel activities, of shape (bn, nv, nt)
    '''
    def __init__(self, incoming, W=lasagne.init.Normal(0.01),  b=lasagne.init.Constant(0.), nonlinearity=None, **kwargs):
        super(pvFWRFLayer, self).__init__(incoming, **kwargs)
        self.nf, self.nv, self.nt = self.input_shape[1:4]
        self.W = self.add_param(W, (self.nf, self.nv, self.nt), name='W')
        if b is not None:
            self.b = self.add_param(b, (1, self.nv, self.nt), name='b', regularizable=False)
            self.b = T.patternbroadcast(self.b, (True, False, False))
        else:
            self.b = None
        self.nonlinearity = (NL.identity if nonlinearity is None else nonlinearity)
        
    def get_output_for(self, input, **kwargs):
        _pred = T.batched_tensordot(input.flatten(ndim=3).dimshuffle((2,0,1)), \
                self.W.flatten(ndim=2).dimshuffle((1,0)), axes=[[2],[1]]) \
                .dimshuffle((1,0)).reshape((input.shape[0],self.nv,self.nt))
        if self.b is not None:
            _pred = _pred + self.b
        return self.nonlinearity(_pred)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nv, self.nt)
    



class svFWRFLayer(L.Layer):
    '''
    svFWRFLayer is a new lasagne layer for 'shared voxel (sv)' candidate receptive field models. It assumes an input
    of shape (bn, nf, bt) where bn is a batch of the time series, nf are the total number of features, bv is a batch of voxels
    and bt is a batch of candidate rf.

    The return values correspond to the predicted voxel activities, of shape (bn, nv, nt)
    '''
    def __init__(self, incoming, nvoxels, W=lasagne.init.Normal(0.01),  b=lasagne.init.Constant(0.), nonlinearity=None, **kwargs):
        super(svFWRFLayer, self).__init__(incoming, **kwargs)
        self.nf = self.input_shape[1]
        self.nt = self.input_shape[2]
        self.nv = nvoxels
        self.W = self.add_param(W, (self.nf, self.nv, self.nt), name='W')
        if b is not None:
            self.b = self.add_param(b, (1, self.nv, self.nt), name='b', regularizable=False)
            self.b = T.patternbroadcast(self.b, (True, False, False))
        else:
            self.b = None
        self.nonlinearity = (NL.identity if nonlinearity is None else nonlinearity)
        
    def get_output_for(self, input, **kwargs):
        _pred = T.batched_tensordot(input.dimshuffle((2,0,1)), self.W.dimshuffle((2,0,1)), axes=[[2],[1]]).dimshuffle((1,2,0))
        if self.b is not None:
            _pred = _pred + self.b
        return self.nonlinearity(_pred)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nv, self.nt)





def get_mst_data(__fmaps, __smsts): 
    '''Apply a tentative fwrf model of the classification network intermediary representations.
    _fmaps is a list of grouped feature maps at different resolutions. F maps in total.
    _smsts is a matching resolution stack of batch_t RF model candidates.
    returns a symbolic tensor of receptive field candiate weighted feature maps (bn, features, bv, bt)'''
    __mstfmaps = [T.tensordot(_fm, __smsts[i], [[2,3], [2,3]])  for i,_fm in enumerate(__fmaps)]
    __mst_data = T.concatenate(__mstfmaps, axis=1)
    return __mst_data


def normalize_mst_data(__mst_data, avg, std):
    _sAvg = theano.shared(avg.T.astype(fpX)[np.newaxis,:,:,np.newaxis])
    _sStd = theano.shared(std.T.astype(fpX)[np.newaxis,:,:,np.newaxis])
    ### set the broadcastability of the sample axis
    _sAvg = T.patternbroadcast(_sAvg, (True, False, False, False))
    _sStd = T.patternbroadcast(_sStd, (True, False, False, False))
    return (__mst_data - _sAvg) / _sStd, [_sAvg, _sStd]


def pvFWRF(__mst_data, nf, nv, nt): 
    '''
    Create a symbolic lasagne network for the per voxel candidate case.
    returns a symbolic outpuy of shape (bn, bv, bt).
    '''
    _input = L.InputLayer((None, nf, nv, nt), input_var=__mst_data.reshape((-1,nf,nv,nt)))
    ## try to add a parametrized local nonlinearity layer.
    _pred  = pvFWRFLayer(_input, W=I.Normal(0.02), b=I.Constant(0.), nonlinearity=None)
    #print "> input using approx %.1f x batch_size Mb of memory (VRAM and RAM)" % (fpX(4*nf*nv*nt) /(1024*1024))
    #print "> output using approx %.3f x batch_size Mb of memory (VRAM and RAM)" % (fpX(4*nv*nt) /(1024*1024))
    return _pred


def svFWRF(__mst_data, nf, nv, nt): 
    '''
    Create a symbolic lasagne network for the shared voxel candidate case.
    returns a symbolic outpuy of shape (bn, bv, bt).
    '''
    _input = L.InputLayer((None, nf, nt), input_var=__mst_data.reshape((-1,nf,nt)))
    _pred  = svFWRFLayer(_input, nvoxels=nv, W=I.Normal(0.02), b=I.Constant(0.), nonlinearity=None) #NL.tanh
    #print "> input using approx %.1f x batch_size Mb of memory (VRAM and RAM)" % (fpX(4*nf*nv*nt) /(1024*1024))
    #print "> output using approx %.3f x batch_size Mb of memory (VRAM and RAM)" % (fpX(4*nv*nt) /(1024*1024))
    return _pred



########################################################################
###              THE MAIN MODEL FUNCTION                             ###
########################################################################

def svModelSpace(sharedModel_specs):
    vm = np.asarray(sharedModel_specs[0])
    nt = np.prod([sms.length for sms in sharedModel_specs[1]])           
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(sharedModel_specs[1])]
    xs, ys, ss = np.meshgrid(rx, ry, rs, indexing='ij')
    return xs.reshape((1,nt)).astype(dtype=fpX), ys.reshape((1,nt)).astype(dtype=fpX), ss.reshape((1,nt)).astype(dtype=fpX) 


def model_space_tensor(
        datas, sharedModel_specs, _symbolicFeatureMaps=None, featureMapSizes=None, _symbolicInputVars=None, 
        nonlinearity=None, zscore=False, mst_avg=None, mst_std=None, epsilon=1e-6, trn_size=None,
        batches=(1,1), view_angle=20., verbose=False, dry_run=False):
    '''
    batches dims are (samples, candidates)

    Feature maps can be provided symbolically, in which case the input will be something else from which the feature maps would be
    calculated and a symbolic variable has to be provided to represent the input. If the feature maps are directly the input, then all 
    these symbols are created automatically.

    This function returns a 4 dimensional model_space tensor, which has dimensions (samples, total number of features, 1, total number of candidates rf).
    The singleton dimension represent the voxels index. However in our case, all voxels share the same candidates rf which is why this dimension is 1.
    '''
    n = len(datas[0])
    bn, bt = batches
    vm = np.asarray(sharedModel_specs[0])
    nt = np.prod([sms.length for sms in sharedModel_specs[1]])          
    mx, my, ms = svModelSpace(sharedModel_specs)
    nbt = nt // bt
    rbt = nt - nbt * bt
    assert rbt==0, "the candidate batch size must be an exact divisor of the total number of candidates"
    ### CHOOSE THE INPUT VARIABLES
    print 'CREATING SYMBOLS\n'
    if _symbolicFeatureMaps is None:
        _fmaps, fmap_sizes = [], []
        for d in datas:
            _fmaps += [T.tensor4(),] 
            fmap_sizes += [d.shape,]
    else:
        _fmaps = _symbolicFeatureMaps
        fmap_sizes = featureMapsSizes
        assert fmap_sizes is not None

    if _symbolicInputVars is None:
        _invars = _fmaps
        for d,fs in zip(datas,fmap_sizes):
            assert d.shape[1:]==fs[1:]
    else:
        _invars = _symbolicInputVars
    ### CREATE SYMBOLIC EXPRESSIONS AND COMPILE
    _smsts, nf = create_shared_batched_feature_maps_gaussian_weights(fmap_sizes, 1, bt, verbose=verbose)
    _mst_data = get_mst_data(_fmaps, _smsts)  
    if verbose:
        print ">> Storing the full modelspace tensor will require approx %.03fGb of RAM!" % (fpX(n*nf*nt*4) / 1024**3)
        print ">> Will be divided in chunks of %.03fGb of VRAM!\n" % ((fpX(n*nf*bt*4) / 1024**3))
    print 'COMPILING...'
    sys.stdout.flush()
    comp_t = time.time()
    mst_data_fn  = theano.function(_invars, _mst_data)
    print '%.2f seconds to compile theano functions' % (time.time()-comp_t)
    ### EVALUATE MODEL SPACE TENSOR
    start_time = time.time()
    print "\nPrecomputing mst candidate responses..."
    sys.stdout.flush()
    mst_data = np.ndarray(shape=(n,nf,1,nt), dtype=fpX)   
    if dry_run:
        return mst_data, None, None
    for t in tqdm(range(nbt)): ## CANDIDATE BATCH LOOP     
        # set the receptive field weight for this batch of voxelmodel
        set_shared_batched_feature_maps_gaussian_weights(_smsts, mx[:,t*bt:(t+1)*bt], my[:,t*bt:(t+1)*bt], ms[:,t*bt:(t+1)*bt], size=view_angle)
        for excerpt, size in iterate_slice(0, n, bn):
            args = slice_arraylist(datas, excerpt)  
            mst_data[excerpt,:,:,t*bt:(t+1)*bt] = mst_data_fn(*args)
    full_time = time.time() - start_time
    print "%d mst candidate responses took %.3fs @ %.3f models/s" % (nt, full_time, fpX(nt)/full_time)
    ### OPTIONAL NONLINEARITY
    if nonlinearity:
        print "Applying nonlinearity to modelspace tensor..."
        sys.stdout.flush()
        for rr, rl in tqdm(iterate_slice(0, mst_data.shape[3], bt)): 
            mst_data[:,:,:,rr] = nonlinearity(mst_data[:,:,:,rr])
    ### OPTIONAL Z-SCORING
    if zscore:
        if trn_size==None:
            trn_size = len(mst_data)
        print "Z-scoring modelspace tensor..."
        if mst_avg is not None and mst_std is not None:
            print "Using provided z-scoring values."
            sys.stdout.flush()
            assert mst_data.shape[1:]==mst_avg.shape[1:], "%s!=%s" % (mst_data.shape[1:], mst_avg.shape[1:])
            assert mst_data.shape[1:]==mst_std.shape[1:], "%s!=%s" % (mst_data.shape[1:], mst_avg.shape[1:])
            mst_avg_loc = mst_avg 
            mst_std_loc = mst_std
            for rr, rl in tqdm(iterate_slice(0, mst_data.shape[3], bt)):   
                mst_data[:,:,:,rr] -= mst_avg_loc[:,:,:,rr]
                mst_data[:,:,:,rr] /= mst_std_loc[:,:,:,rr]
                mst_data[:,:,:,rr] = np.nan_to_num(mst_data[:,:,:,rr])        
        else: # calculate the z-score stat the first time around.
            print "Using self z-scoring values."
            sys.stdout.flush()
            mst_avg_loc = np.ndarray(shape=(1,)+mst_data.shape[1:], dtype=fpX)
            mst_std_loc = np.ndarray(shape=(1,)+mst_data.shape[1:], dtype=fpX)
            for rr, rl in tqdm(iterate_slice(0, mst_data.shape[3], bt)):   
                mst_avg_loc[0,:,:,rr] = np.mean(mst_data[:trn_size,:,:,rr], axis=0, dtype=np.float64).astype(fpX)
                mst_std_loc[0,:,:,rr] =  np.std(mst_data[:trn_size,:,:,rr], axis=0, dtype=np.float64).astype(fpX) + fpX(epsilon)
                mst_data[:,:,:,rr] -= mst_avg_loc[:,:,:,rr]
                mst_data[:,:,:,rr] /= mst_std_loc[:,:,:,rr]
                mst_data[:,:,:,rr] = np.nan_to_num(mst_data[:,:,:,rr])
    ### Free the VRAM
    for _s in _smsts:
        _s.set_value(np.asarray([], dtype=fpX).reshape((0,0,0,0)))
    return mst_data, mst_avg_loc, mst_std_loc



def learn_params(
        mst_data, voxels, w_params, \
        batches=(1,1,1), val_test_size=100, lr=1e-4, l2=0.0, num_epochs=1, output_val_scores=-1, output_val_every=1, verbose=False, dry_run=False):
    ''' 
        batches dims are (samples, voxels, candidates)
    '''
    assert len(mst_data)==len(voxels), "data/target length mismatch"  
    n, nf, _, nt = mst_data.shape
    _, nv = voxels.shape
    bn, bv, bt = batches    
    nbv, nbt = nv // bv, nt // bt
    rbv, rbt = nv - nbv * bv, nt - nbt * bt
    assert rbt==0, "the model batch size must be an divisor of the total number of models"
    if verbose:
        print "Grad. Desc. planned in %d batch with batch size %d and residual %d" % \
            (int(np.ceil(float(n-val_test_size) / bn)), bn, (n-val_test_size)%bn)
        print "%d voxel batches of size %d with residual %d" % (nbv, bv, rbv)
        print "%d candidate batches of size %d with residual %d" % (nbt, bt, rbt)
        print "for %d voxelmodel fits." % (nv*nt)
        sys.stdout.flush()     

    print 'CREATING SYMBOLS\n'
    _V  = T.matrix()
    __V = _V.dimshuffle((0,1,'x'))
    __lr = theano.shared(fpX(lr))
    __l2 = theano.shared(fpX(l2))    
    ### request shared memory    
    __mst_sdata = theano.shared(np.zeros(shape=(n, nf, 1, bt), dtype=fpX))
    __vox_sdata = theano.shared(np.zeros(shape=(n, bv), dtype=fpX))
    __range = T.ivector()
    _smst_batch = __mst_sdata[__range[0]:__range[1]]
    _fwrf_o = svFWRF(_smst_batch, nf, bv, bt)
    if verbose:
        plu.print_lasagne_network(_fwrf_o, skipnoparam=False)
    ### define and compile the training expressions.       
    _fwrf_o_reg = __l2 * R.regularize_layer_params(_fwrf_o, R.l2)
    fwrf_o_params = L.get_all_params(_fwrf_o, trainable=True)

    _sV = __vox_sdata[__range[0]:__range[1]].dimshuffle((0,1,'x'))
    _fwrf_o_trn_pred = L.get_output(_fwrf_o, deterministic=False)
    _fwrf_o_trn_preloss = O.squared_error(_fwrf_o_trn_pred, _sV).mean(axis=0)
    _fwrf_o_trn_loss = _fwrf_o_trn_preloss.sum() + _fwrf_o_reg

    _fwrf_o_val_pred = L.get_output(_fwrf_o, deterministic=True)
    _fwrf_o_val_preloss = O.squared_error(_fwrf_o_val_pred, _sV).mean(axis=0) #average across the batch elements
    ###
    __fwrf_o_updates = lasagne.updates.sgd(_fwrf_o_trn_loss, fwrf_o_params, learning_rate=__lr)
    #__fwrf_o_updates = lasagne.updates.adam(_fwrf_o_trn_loss, fwrf_o_params, learning_rate=self.__lr, beta1=0.5, epsilon=1e-12)
    print 'COMPILING...'
    sys.stdout.flush()
    comp_t = time.time()
    fwrf_o_trn_fn = theano.function([__range], updates=__fwrf_o_updates)
    fwrf_o_val_fn = theano.function([__range], _fwrf_o_val_preloss)
    print '%.2f seconds to compile theano functions' % (time.time()-comp_t)

    ### shuffle the time series of voxels and mst_data
    order = np.arange(n, dtype=int)
    np.random.shuffle(order)
    mst_data = mst_data[order]
    voxels = voxels[order]        
        
    ### THIS IS WHERE THE MODEL OPTIMIZATION IS PERFORMED ### 
    print "\nVoxel-Candidates model optimization..."
    start_time = time.time()
    val_batch_scores = np.zeros((bv, bt), dtype=fpX)
    best_epochs = np.zeros(shape=(nv), dtype=int)
    best_scores = np.full(shape=(nv), fill_value=np.inf, dtype=fpX)
    best_models = np.zeros(shape=(nv), dtype=int)

    W, b = fwrf_o_params
    best_w_params = [np.zeros(p.shape, dtype=fpX) for p in w_params]      
        
    ### save score history
    num_outputs = int(num_epochs / output_val_every) + int(num_epochs%output_val_every>0)
    val_scores = []
    if output_val_scores==-1:
        val_scores  = np.zeros(shape=(num_outputs, nv, nt), dtype=fpX) 
    elif output_val_scores>0:
        outv = output_val_scores
        val_scores  = np.zeros(shape=(num_outputs, bv*outv, nt), dtype=fpX) 
    ###
    if dry_run:
        __mst_sdata.set_value(np.asarray([], dtype=fpX).reshape((0,0,0,0)))
        __vox_sdata.set_value(np.asarray([], dtype=fpX).reshape((0,0)))
        W.set_value(np.asarray([], dtype=fpX).reshape((0,)*len(W.get_value().shape)))
        b.set_value(np.asarray([], dtype=fpX).reshape((0,)*len(b.get_value().shape)))
        return val_scores, best_scores, best_epochs, best_models, best_w_params
    ### VOXEL LOOP
    for v, (rv, lv) in tqdm(enumerate(iterate_range(0, nv, bv))):
        voxelSlice = voxels[:,rv]
        best_epochs_slice = best_epochs[rv] 
        best_scores_slice = best_scores[rv]
        best_models_slice = best_models[rv] 
        rW, rb = w_params[0][rv,:], w_params[1][rv]
        if lv<bv: #PATCH UP MISSING DATA FOR THE FIXED VOXEL BATCH SIZE
            voxelSlice = np.concatenate((voxelSlice, np.zeros(shape=(n, bv-lv), dtype=fpX)), axis=1)
            rW = np.concatenate((rW, np.zeros(shape=(bv-lv, nf), dtype=fpX)), axis=0)
            rb = np.concatenate((rb, np.zeros(shape=(bv-lv), dtype=fpX)), axis=0)       
        pW = np.repeat(rW.T, repeats=bt).reshape((nf,bv,bt)) # ALL CANDIDATE MODELS GET THE SAME INITIAL PARAMETER VALUES
        pb = np.repeat(rb, repeats=bt).reshape((1, bv,bt))      
                    
        set_shared_parameters([__vox_sdata], [voxelSlice])
        ### CANDIDATE LOOP
        for t in range(nbt): ## CANDIDATE BATCH LOOP
            # need to recompile to reset the solver!!! (depending on the solver used)
            fwrf_o_trn_fn = theano.function([__range], updates=__fwrf_o_updates)
            # set the shared parameter values for this candidates. Every candidate restart at the same point.
            set_shared_parameters(fwrf_o_params+[__mst_sdata], [pW, pb, mst_data[:,:,:,t*bt:(t+1)*bt]])
            print "\n  Voxel %d:%d of %d, Candidate %d:%d of %d" % (rv[0], rv[-1]+1, nv, t*bt, (t+1)*bt, nt)
            ### EPOCH LOOP
            epoch_start = time.time()
            for epoch in range(num_epochs):
                ######## ONE EPOCH OF TRAINING ###########
                val_batch_scores.fill(0)  
                # In each epoch, we do a full pass over the training data:
                for rb, lb in iterate_bounds(0, n-val_test_size, bn):
                    fwrf_o_trn_fn(rb)
                # and one pass over the validation set.  
                val_batches = 0
                for rb, lb in iterate_bounds(n-val_test_size, val_test_size, bn): 
                    loss = fwrf_o_val_fn(rb)
                    val_batch_scores += loss
                    val_batches += lb
                val_batch_scores /= val_batches
                if verbose:
                    print "    validation <loss>: %.6f" % (val_batch_scores.mean())
                ### RECORD TIME SERIES ###
                if epoch%output_val_every==0:
                    if output_val_scores==-1:
                        val_scores[int(epoch / output_val_every), rv, t*bt:(t+1)*bt] = val_batch_scores[:lv,:] 
                    elif output_val_scores>0:
                        val_scores[int(epoch / output_val_every), v*outv:(v+1)*outv, t*bt:(t+1)*bt] = val_batch_scores[:min(outv, lv),:]
                ##### RECORD MINIMUM SCORE AND MODELS #####
                best_models_for_this_epoch = np.argmin(val_batch_scores[:lv,:], axis=1)
                best_scores_for_this_epoch = np.amin(val_batch_scores[:lv,:], axis=1)
                # This updates the BEST RELATIVE MODELS, along with their associated scores 
                best_scores_mask = (best_scores_for_this_epoch<best_scores_slice) #all the voxels that show an improvement
                best_epochs_slice[best_scores_mask] = epoch  
                np.copyto(best_scores_slice, best_scores_for_this_epoch, casting='same_kind', where=best_scores_mask)      
                np.copyto(best_models_slice, best_models_for_this_epoch + t*bt, casting='same_kind', where=best_scores_mask) #notice the +t*bt to return the best model across all models, not just the batch's
                #to select the weight slices we need, we need to specify the voxels that showed improvement AND the models that correspond to these improvements.
                update_vm_pos = np.zeros((bv, bt), dtype=bool)
                update_vm_pos[np.arange(lv)[best_scores_mask], best_models_for_this_epoch[best_scores_mask]] = True
                update_vm_idx = np.arange(bv*bt)[update_vm_pos.flatten()]
                # update the best parameter values based on the voxelmodel validation scores.
                best_w_params[0][np.asarray(rv)[best_scores_mask], :] = (W.get_value().reshape((nf,-1))[:,update_vm_idx]).T
                best_w_params[1][np.asarray(rv)[best_scores_mask]]    = b.get_value().reshape((-1))[update_vm_idx]   

            batch_time = time.time()-epoch_start
            print "    %d Epoch for %d voxelmodels took %.3fs @ %.3f voxelmodels/s" % (num_epochs, lv*bt, batch_time, fpX(lv*bt)/batch_time)
            sys.stdout.flush()
        #end candidate loop    
        best_epochs[rv] = np.copy(best_epochs_slice)
        best_scores[rv] = np.copy(best_scores_slice) ##NECESSARY TO COPY BACK
        best_models[rv] = np.copy(best_models_slice)   
    # end voxel loop 
    # free shared vram
    __mst_sdata.set_value(np.asarray([], dtype=fpX).reshape((0,0,0,0)))
    __vox_sdata.set_value(np.asarray([], dtype=fpX).reshape((0,0)))
    W.set_value(np.asarray([], dtype=fpX).reshape((0,)*len(W.get_value().shape)))
    b.set_value(np.asarray([], dtype=fpX).reshape((0,)*len(b.get_value().shape)))

    full_time = time.time() - start_time
    print "\n---------------------------------------------------------------------"
    print "%d Epoch for %d voxelmodels took %.3fs @ %.3f voxelmodels/s" % (num_epochs, nv*nt, full_time, fpX(nv*nt)/full_time)
    return val_scores, best_scores, best_epochs, best_models, best_w_params
    



def get_prediction(mst_data, voxels, mst_rel_models, w_params, batches=(1,1)):
    '''
    batches dims are (samples, voxels)

    Arguments:
     mst_data: The modelspace tensor of the validation set.
     voxels: The corresponding expected voxel response for the validation set.
     mst_rel_models: The relative rf model that have been selected through training.
     params: The trained parameter values of the model.
        
    Returns:
     voxel prediction, per voxel corr_coeff
    '''
    n, nf, _, nt = mst_data.shape
    _, nv = voxels.shape
    bn, bv = batches
    nbv = nv // bv
    rbv = nv - nbv * bv
    assert len(mst_data)==len(voxels)
    assert len(mst_rel_models)==nv ## voxelmodels interpreted as relative model  
    assert mst_rel_models.dtype==int
    assert n<=bn, "validation needs to be done in a single batch."
    print "%d voxel batches of size %d with residual %d" % (nbv, bv, rbv) 

    print 'CREATING SYMBOLS\n'
    _V  = T.matrix()
    __V = _V.dimshuffle((0,1,'x'))
    _mst_data = T.tensor4()
    _fwrf_t = pvFWRF(_mst_data, nf, bv, 1)   
    fwrf_t_params = L.get_all_params(_fwrf_t, trainable=True)
        
    _fwrf_t_val_pred = L.get_output(_fwrf_t, deterministic=True)   
    _fwrf_t_val_cc = ((_fwrf_t_val_pred - _fwrf_t_val_pred.mean(axis=0, keepdims=True)) * (__V - __V.mean(axis=0, keepdims=True))).mean(axis=0) / \
        T.sqrt(T.sqr(_fwrf_t_val_pred - _fwrf_t_val_pred.mean(axis=0, keepdims=True)).mean(axis=0) * T.sqr(__V - __V.mean(axis=0, keepdims=True)).mean(axis=0))         
    print 'COMPILING...'
    sys.stdout.flush()
    comp_t = time.time()
    fwrf_t_pred_fn = theano.function([_mst_data], _fwrf_t_val_pred)
    fwrf_t_test_fn = theano.function([_mst_data, _V], [_fwrf_t_val_pred, _fwrf_t_val_cc])        
    print '%.2f seconds to compile theano functions' % (time.time()-comp_t)

    predictions = np.zeros(shape=(n, nv), dtype=fpX)
    cc_scores   = np.zeros(shape=(nv), dtype=fpX)
    ### VOXEL BATCH LOOP
    for rv, lv in tqdm(iterate_range(0, nv, bv)):
        voxelSlice = voxels[:,rv]
        vm_slice = mst_rel_models[rv]
        rW = w_params[0][rv,:]
        rb = w_params[1][rv]
        if lv<bv: #PATCH UP MISSING DATA FOR THE FIXED BATCH SIZE
            voxelSlice = np.concatenate((voxelSlice, np.zeros(shape=(n, bv-lv), dtype=fpX)), axis=1)
            vm_slice = np.concatenate((vm_slice, np.zeros(shape=(bv-lv), dtype=int)), axis=0)
            rW = np.concatenate((rW, np.zeros(shape=(bv-lv, nf), dtype=fpX)), axis=0)
            rb = np.concatenate((rb, np.zeros(shape=(bv-lv), dtype=fpX)), axis=0) 
        pW = rW.T.reshape((nf,bv,1))
        pb = rb.reshape((1,bv,1))      

        pv_mst_data = mst_data[:, :, 0, vm_slice, np.newaxis]
        set_shared_parameters(fwrf_t_params, [pW, pb])
        ###            
        pred, cc = fwrf_t_test_fn(pv_mst_data, voxelSlice)
        predictions[:, rv], cc_scores[rv] = pred[:,:lv,0], cc[:lv,0]
    return predictions, cc_scores



def real_space_model(mst_rel_models, sharedModel_specs, mst_avg=None, mst_std=None):
    '''
    Convert candidate in the model space tensor into real space, per-voxel models.
    '''
    nv = len(mst_rel_models)
    vm = np.asarray(sharedModel_specs[0])
    nt = np.prod([sms.length for sms in sharedModel_specs[1]])         
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(sharedModel_specs[1])] # needed to map rf's back to visual space

    best_rf_params = rel_to_abs_shared_models(mst_rel_models, rx, ry, rs) ### put back the models in absolute coordinate a.k.a model spec for the next iteration

    best_mst_data_avg = None
    best_mst_data_std = None
    if mst_avg is not None and mst_std is not None:
        nf = mst_avg.shape[1]
        best_mst_data_avg = np.ndarray(shape=(nv, nf), dtype=fpX)
        best_mst_data_std = np.ndarray(shape=(nv, nf), dtype=fpX)
        for v in range(nv):
            best_mst_data_avg[v,:] = mst_avg[0,:,0,mst_rel_models[v]]
            best_mst_data_std[v,:] = mst_std[0,:,0,mst_rel_models[v]]
    return best_rf_params, best_mst_data_avg, best_mst_data_std



def get_symbolic_prediction(_symbolicFeatureMaps, featureMapSizes, rf_params, w_params, avg=None, std=None, nonlinearity=None, view_angle=20.0):
    '''
    Unlike the training procedure which is trained by part, this creates a matching theano expression from end-to-end.
    
    Arguments:
    _symbolicFeatureMaps: the symbolic feature maps using for training
    featureMapSizes: the feature maps sizes
    voxelmodels: the absolute receptive field coordinate i.e. a (V,3) array whose entry are (x, y, sigma)
    params: the feature tuning parameters
    (optional) avg, std: the average and standard deviation from z-scoring.
    (optional) nonlinearity: a callable function f(x) which returns a theano expression for an elementwise nonlinearity.
    view_angle (default 20.0): Same as during training. This just fix the scale relative to the values of the voxelmodels.
    
    Returns:
    A symbolic variable representing the prediction of the fwRF model.
    A dictionary of all the shared variables.
    
    Note: There is quite a bit of repetition due the idiosyncracies of the training procedure. Make sure
    that this compiled expression returns the same values has validate_models when run on the same data.
    '''
    shared_var = {}
    nf = np.sum([fm[1] for fm in featureMapSizes])
    nv = rf_params.shape[0]
    assert rf_params.shape[1]==3

    print 'CREATING SYMBOLS\n'
    _smsts,_ = create_shared_batched_feature_maps_gaussian_weights(featureMapSizes, nv, 1, verbose=True)
    shared_var['fpf_weight'] = _smsts
    if avg is not None:
        if nonlinearity is not None:
            _nmst, _stats = normalize_mst_data(nonlinearity(mst_data(_symbolicFeatureMaps, _smsts)), avg, std)
            _fwrf = pvFWRF(_nmst, nf, nv, 1)
        else:
            _nmst, _stats = normalize_mst_data(mst_data(_symbolicFeatureMaps, _smsts), avg, std)
            _fwrf = pvFWRF(_nmst, nf, nv, 1)
        shared_var['mst_norm'] = _stats
    else:
        if nonlinearity is not None:
            _fwrf = pvFWRF(nonlinearity(mst_data(_symbolicFeatureMaps, _smsts)), nf, nv, 1)
        else:
            _fwrf = pvFWRF(mst_data(_symbolicFeatureMaps, _smsts), nf, nv, 1)            
    plu.print_lasagne_network(_fwrf, skipnoparam=False)
        
    fwrf_params = L.get_all_params(_fwrf, trainable=True)
    shared_var['fwrf_params'] = fwrf_params
    set_shared_parameters(fwrf_params, [w_params[0].T.reshape((nf,nv,1)), w_params[1].reshape((1,nv,1))])
    set_shared_batched_feature_maps_gaussian_weights(_smsts, rf_params[:,0], rf_params[:,1], rf_params[:,2], size=view_angle)      
        
    return L.get_output(_fwrf, deterministic=True).flatten(ndim=2), shared_var


################################################################
###                 K-OUT VARIANTS                           ###
################################################################
def kout_learn_params(mst_data, voxels, val_sample_order, w_params, batches=(1,1,1), val_part_size=1, holdout_size=1, lr=1e-4, l2=0.0, num_epochs=1, verbose=False, dry_run=False, test_run=False):
    '''
        A k-out variant of the fwrf shared_model_training routine.

        batches dims are (samples, voxels, candidates)
    '''
    data_size, nv = voxels.shape
    num_val_part = int(data_size / val_part_size)
    trn_size = data_size - val_part_size

    assert np.modf(float(data_size)/val_part_size)[0]==0.0, "num_val_part (%d) has to be an exact divisor of the set size (%d)" % (num_val_part, data_size)
    print "trn_size = %d (incl. holdout), holdout_size = %d, val_size = %d\n" % (trn_size, holdout_size, val_part_size)
    model = {}
    if test_run:
        tnv = batches[1]
        print "####################################"
        print "### Test run %d of %d voxels ###" % (tnv, nv)
        print "####################################"       
        k, (vs,ls) = 0, (slice(0, val_part_size), val_part_size)
        
        trn_mask = np.ones(data_size, dtype=bool)
        trn_mask[val_sample_order[vs]] = False # leave out the first batch of validation point
            
        trn_mst_data = mst_data[trn_mask]
        val_mst_data = mst_data[~trn_mask]

        trn_voxel_data = voxels[trn_mask, 0:tnv]
        val_voxel_data = voxels[~trn_mask, 0:tnv]
        voxelParams = [p[0:tnv] for p in w_params]
        ### fit this part ###
        val_scores, best_scores, best_epochs, best_candidates, best_w_params = learn_params(\
            trn_mst_data, trn_voxel_data, w_params, batches=batches,\
            val_test_size=holdout_size, lr=lr, l2=l2, num_epochs=num_epochs, output_val_scores=-1, output_val_every=1, verbose=verbose, dry_run=dry_run)
        val_pred, val_cc = get_prediction(val_mst_data, val_voxel_data, best_candidates, best_w_params, batches=(val_part_size, batches[1]))

        model[k] = {}
        model[k]['val_scores'] = val_scores 
        model[k]['scores']    = best_scores
        model[k]['epochs']    = best_epochs
        model[k]['w_params']  = best_w_params
        model[k]['candidates'] = best_candidates
        model[k]['val_mask']  = ~trn_mask
        model[k]['val_cc']    = val_cc    
        #####################
        model['n_parts'] = 1       
        model['val_pred'] = val_pred
        model['val_cc'] = val_cc
       
    else:
        # The more parts, the more data each part has to learn the prediction. It's a leave k-out.
        full_val_pred = np.zeros(shape=voxels.shape, dtype=fpX)
        for k,(vs,ls) in enumerate(iterate_slice(0, data_size, val_part_size)):
            print "################################"
            print "###   Resampling block %2d   ###" % k
            print "################################"
            trn_mask = np.ones(data_size, dtype=bool)
            trn_mask[val_sample_order[vs]] = False # leave out the first batch of validation point
            
            trn_mst_data = mst_data[trn_mask]
            val_mst_data = mst_data[~trn_mask]

            trn_voxel_data = voxels[trn_mask]
            val_voxel_data = voxels[~trn_mask]
            ### fit this part ###
            val_scores, best_scores, best_epochs, best_candidates, best_w_params = learn_params(\
                trn_mst_data, trn_voxel_data, w_params, batches=batches,\
                val_test_size=holdout_size, lr=lr, l2=l2, num_epochs=num_epochs, output_val_scores=0, output_val_every=10, verbose=verbose, dry_run=dry_run)
            val_pred, val_cc = get_prediction(val_mst_data, val_voxel_data, best_candidates, best_w_params, batches=(val_part_size, batches[1]))

            model[k] = {}
            model[k]['scores']    = best_scores
            model[k]['epochs']    = best_epochs
            model[k]['w_params']  = best_w_params
            model[k]['candidates'] = best_candidates
            model[k]['val_mask']  = ~trn_mask
            model[k]['val_cc']    = val_cc    
            #####################
            full_val_pred[~trn_mask] = val_pred 
        ##
        full_cc = np.zeros(nv)
        for v in range(nv):
            full_cc[v] = np.corrcoef(full_val_pred[:,v], voxels[:,v])[0,1]
        ## global pred and cc
        model['n_parts'] = num_val_part
        model['val_pred'] = full_val_pred
        model['val_cc'] = full_cc
    return model


def kout_get_prediction(mst_data, voxels, model, batches=(1,1)):
    '''
        A k-out variant of the fwrf get_prediction(...) routine
    '''
    data_size, nv = voxels.shape
    num_val_part = model['n_parts']
    assert np.prod([k in model.keys() for k in range(num_val_part)])>0
    full_val_pred = np.zeros(shape=voxels.shape, dtype=fpX)
    for k in range(num_val_part):
        print "################################"
        print "###   Resampling block %2d   ###" % k
        print "################################"

        val_mask = model[k]['val_mask']
        best_w_params   = model[k]['w_params']
        best_candidates = model[k]['candidates']
        
        val_mst_data = mst_data[val_mask]
        val_voxel_data = voxels[val_mask]

        val_pred,_ = get_prediction(val_mst_data, val_voxel_data, best_candidates, best_w_params, batches=batches)
        full_val_pred[val_mask] = val_pred 
    full_cc = np.zeros(nv)
    for v in range(nv):
        full_cc[v] = np.corrcoef(full_val_pred[:,v], voxels[:,v])[0,1]
    return full_val_pred, full_cc


def kout_real_space_model(model, sharedModel_specs, mst_avg=None, mst_std=None):
    '''
        A k-out variant of the fwrf real_space_model(...) routine
    '''
    num_val_part = model['n_parts']
    assert np.prod([k in model.keys() for k in range(num_val_part)])>0
    for k in range(num_val_part):
        mst_rel_models = model[k]['candidates']

        best_rf_params, best_mst_data_avg, best_mst_data_std = real_space_model(mst_rel_models, sharedModel_specs, mst_avg=mst_avg, mst_std=mst_std)
        model[k]['rf_params'] = best_rf_params
        model[k]['norm_avg'] = best_mst_data_avg
        model[k]['norm_std'] = best_mst_data_std
    return model

