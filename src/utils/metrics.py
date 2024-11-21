import scipy
import numpy as np
import cv2




eps = 1e-7 #regularization value

def g_filter(shape =(200,200), sigma=60):
    """
    Using Gaussian filter to generate center bias
    """
    x, y = [edge /2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-int(x), int(x))] for j in range(-int(y), int(y))])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    return g_filter


def add_center_bias(salMap):
    cb = g_filter()
    cb = cv2.resize(cb,(salMap.shape[1],salMap.shape[-2]),interpolation = cv2.INTER_LINEAR)
    salMap = salMap*cb
    return salMap

def cal_cc_score(salMap, fixMap):
    """
    Compute CC score between two attention maps
    """
    salMap /= np.sum(salMap)
    fixMap = fixMap/np.sum(fixMap)
    score = np.corrcoef(salMap.reshape(-1), fixMap.reshape(-1))[0][1]

    return score




def cal_sim_score(salMap, fixMap):
    """
    Compute SIM score between two attention maps
    """
    salMap = salMap/np.sum(salMap)
    fixMap = fixMap/np.sum(fixMap)

    sim_score = np.sum(np.minimum(salMap,fixMap))

    return sim_score


def cal_kld_score(salMap,fixMap): #recommand salMap to be free-viewing attention
    """
    Compute KL-Divergence score between two attention maps
    """
    salMap = salMap/np.sum(salMap)
    fixMap = fixMap/np.sum(fixMap)
    kl_score = fixMap*np.log(eps+fixMap/(salMap+eps))
    kl_score = np.sum(kl_score)

    return kl_score


def cal_auc_score(salMap, fixMap):
    """
    compute the AUC score for saliency prediction
    """
    salMap /= salMap.max()
    fixmap = (fixMap==1).astype(int)
    salShape = salMap.shape
    fixShape = fixmap.shape

    predicted = salMap.reshape(salShape[0]*salShape[1], -1,
                               order='F').flatten()
    actual = fixmap.reshape(fixShape[0]*fixShape[1], -1,
                            order='F').flatten()
    labelset = np.arange(2)

    auc = area_under_curve(predicted, actual, labelset)
    return auc

def area_under_curve(predicted, actual, labelset):
    tp, fp = roc_curve(predicted, actual, np.max(labelset))
    auc = auc_from_roc(tp, fp)
    return auc

def auc_from_roc(tp, fp):
    h = np.diff(fp)
    auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
    return auc

def roc_curve(predicted, actual, cls):
    si = np.argsort(-predicted)
    tp = np.cumsum(np.single(actual[si]==cls))
    fp = np.cumsum(np.single(actual[si]!=cls))
    tp = tp/np.sum(actual==cls)
    fp = fp/np.sum(actual!=cls)
    tp = np.hstack((0.0, tp, 1.0))
    fp = np.hstack((0.0, fp, 1.0))
    return tp, fp

def cal_nss_score(salmap,fixmap,center_bias=False):
    #compute the normalized scanpath saliency
    salmap = (salmap-np.mean(salmap))/np.std(salmap)
    
    return np.sum(salmap * fixmap)/np.count_nonzero(fixmap)
