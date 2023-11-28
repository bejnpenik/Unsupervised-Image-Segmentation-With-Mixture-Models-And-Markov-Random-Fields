import numpy as np
import yaml
import skimage.io as io
import skimage.util as util
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import njit, prange
from numba import cuda
from numba import types
from tqdm import tqdm
from enum import Enum, unique, auto
from math import pi, exp, log, lgamma, isfinite, sqrt
from sys import float_info
import math
from yaml import load, dump
import os
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import argparse
from utils import *


def estimate_true_components_from_segmentation(target, X, Y, force = "nearest"):
    results = []
    print("Finding clusters in full segmentation map...")
    _,_,z = X.shape
    for i in tqdm(range(z)):
        x,y = X[:,:, i], Y[:,:,i]
        uniques, counts = np.unique(x, return_counts=True)
        for u,c in zip(uniques, counts):
            mask = y[x==u]
            minv,maxv,medianv,meanv = np.min(mask), np.max(mask), np.median(mask), np.mean(mask)
            results.append((u,minv,maxv,medianv,meanv,c))
    results = np.array(results)
    final = []
    for c in np.unique(results[:, 0]):
        mask = results[:,0]==c
        minv, maxv = np.min(results[mask,1]), np.max(results[mask,2])
        medianv, meanv = np.median(results[mask,3]), np.mean(results[mask,4])
        count = np.sum(results[mask, 5])
        final.append([c, minv, maxv, medianv, meanv, count])
    
    results = np.array(final)
    results = results[np.argsort(results[:, 3]), ]
    
    predict = 0
    i = 0
    while predict < target:
        predict += results[i, 5]
        i = i + 1
        
        
    score_above = abs(predict/target-1)
    predict_last = predict - results[i-1, 5]
    score_bellow = abs(predict_last/target-1)
    
    if force == "nearest":
        if score_above<score_bellow:
            iters = i
            setting = "score above"
        else:
            iters = i - 1
            setting = "score bellow"
            
    elif force == "above":
        iters = i
        setting = "score above"
    elif force == "bellow":
        iters = i - 1
        setting = "score bellow"
    else:
        raise Exception(f"Keyword force set to invalid value: {force}")
    
    
    print(f"Score bellow is {score_bellow} and score above is {score_above}")
    print(f"Setting {setting}")
    
    
    T,F = [],[]
    for _i in range(iters):
       T.append(int(results[_i,0]))
    
    for _r in results:
        c = int(_r[0])
        if c not in T:
           F.append(c)
    
    return T,F

def merge_D(D, T, F):
    _D = np.zeros((D.shape[0], 2), dtype=np.float64)
    for i in T:
        _D[:,1] += D[:, i]
    for i in F:
        _D[:,0] += D[:, i]
    return _D

def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for unsupervised image segmentation of mCT scanned cellular materials")
    parser.add_argument("--cwd", type=str, help="current working directory", default=os.getcwd())
    parser.add_argument("--images-dir", type=str, help="scanned image directory", default="images/")
    parser.add_argument("--pdf", type=str, help="mixture model probability density function", default="normal")
    parser.add_argument("--cmax", type=int, help="maximum number of components in mixture model", default=64)
    parser.add_argument("--cmin", type=int, help="minimum number of components in mixture model", default=1)
    parser.add_argument("--target-pixel-number", type=int, help="number of pixels in porous structure")
    parser.add_argument("--merge-clusters-rounding", type=str, help="One of upper, lower or nearest", default="nearest")
    parser.add_argument("--beta", type=float, help="beta parameter for Markov random fields", default=1)
    parser.add_argument("--nbr-of-icm-iters", type=int, help="number of iterations in Iterated conditional modes algorithm", default=1)
    parser.add_argument("--save-dir", type=str, help="save dir for labeled images", default="labels/")
    return vars(parser.parse_args())


def main(args):
    cwd = args["cwd"]
    img_dir = args["images_dir"]
    pdf = args["pdf"]
    cmax = args["cmax"]
    cmin = args["cmin"]
    target = args["target_pixel_number"]
    merging_type = args["merge_clusters_rounding"]
    beta = args["beta"]
    icm_iters = args["nbr_of_icm_iters"]
    save_path = args["save_dir"]

    if not numba.cuda.is_available():
        raise NotImplementedError("Numba with cuda support should be installed!")
    
    os.chdir(cwd)
    if "model.estimation.R" not in os.listdir("./"):
        raise FileNotFoundError(f"File model.estimation.R should be in {cwd}")
    R_run_script = f"Rscript model.estimation.R --cwd {cwd} --images-dir {img_dir} --cmax {cmax} --cmin {cmin} --pdf {pdf}"
    os.system(R_run_script)

    D = np.loadtxt("D.txt")
    #print(D.shape)
    D_cuda = cuda.to_device(D)
    #target = 66384478

    nop = len(os.listdir(img_dir))
    sx,sy = io.imread(f"{img_dir}/{os.listdir(img_dir)[20]}").shape
    structure_size = (sx,sy,nop)
    print(f"Size of structure is {structure_size}")
    
    Y = np.zeros(structure_size, dtype=np.uint8)
    L = np.zeros(structure_size, dtype=np.uint8)
    j = 0
    print("Estimating full segmentation maps...")
    for i in tqdm(os.listdir(img_dir)):
        if i.endswith(".bmp"):
            img = io.imread(f"{img_dir}/{i}")
            _,_e = i.split(".")
            _ = int(_)
            Y[:,:,int(_-1)] = img
            Yp = cuda.to_device(np.array(img, dtype=np.uint8))
            Xp = cuda.to_device(np.zeros(img.shape, dtype=np.uint8))
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(Yp.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(Yp.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            SEGMENTATION_INITIAL_MAT[blockspergrid, threadsperblock](Yp, Xp, D_cuda)
            L[:,:,int(_-1)] = Xp.copy_to_host()
            j+=1
            del Yp, Xp
    if j < nop:
        print("Number of planes smaller!")
        print("Changing...")
        nop = j
        structure_size = (sx,sy,nop) 

    T, F = estimate_true_components_from_segmentation(target,L,Y,merging_type)
    print(f"True clusters are: ({' '.join(map(str,T))})")
    print(f"False clusters are: ({' '.join(map(str,F))})")
    merged_D = merge_D(D,T,F)
    print(merged_D.shape)
    merged_D_cuda = cuda.to_device(merged_D)

    Y = np.zeros(structure_size, dtype=np.uint8)
    X = np.zeros(structure_size, dtype=np.uint8)
    j = 0

    print("Estimating initial segmentation maps...")
    for i in tqdm(os.listdir(img_dir)):
        if i.endswith(".bmp"):
            img = io.imread(f"{img_dir}/{i}")
            _,_e = i.split(".")
            _ = int(_)
            Y[:,:,int(_-1)] = img
            Yp = cuda.to_device(np.array(img, dtype=np.uint8))
            Xp = cuda.to_device(np.zeros(img.shape, dtype=np.uint8))
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(Yp.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(Yp.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            SEGMENTATION_INITIAL_MAT[blockspergrid, threadsperblock](Yp, Xp, merged_D_cuda)
            X[:,:,int(_-1)] = Xp.copy_to_host()
            j+=1
            del Yp, Xp

    save_path = f"labels/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    _,_,z = X.shape    
    for i in tqdm(range(z)):
        x = util.img_as_ubyte(X[:,:,i]*255)
        io.imsave(f"{save_path}{i}.bmp", x)

    print(f"Estimated target per cent: {np.sum(X)/target}")
    print("Ready for ICM loop")
    
    U1_cuda = cuda.to_device(np.zeros((sx, sy, 2), dtype=np.float32))
    U2_cuda = cuda.to_device(np.zeros((sx, sy, 2), dtype=np.float32))
    Y_cuda = cuda.to_device(Y)
    X_cuda = cuda.to_device(X)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(Y.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(Y.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    Xn = np.zeros(structure_size, dtype=np.uint8)
    res_str = [["T", "E", "niters", "Converged" "Comment"]]   
    #B_cuda = cuda.to_device(B)
    n_iter = icm_iters
    E_old=np.inf
    T_old = np.inf
    for it in range(n_iter):
        print(f"ICM iteration: {it+1}")
        E = 0
        for z in tqdm(range(nop)):
            ICM_MAT_3D_SLICE_SIMPLE[blockspergrid, threadsperblock](Y_cuda, X_cuda, z, 
                                                             U1_cuda, U2_cuda, 
                                                             merged_D_cuda)
            
            U1 = U1_cuda.copy_to_host()
            U2 = U2_cuda.copy_to_host()
            U = U1 + float(beta)*U2 
            Xn[:,:, z] = np.argmin(U, axis=2)
            E += np.sum(U.min(axis=2), dtype=np.float64)
            U1_cuda = cuda.to_device(np.zeros((sx, sy, 2), dtype=np.float32))
            U2_cuda = cuda.to_device(np.zeros((sx, sy, 2), dtype=np.float32))
        T = np.sum(Xn)/target
        print(f"Loop results is {T}")
        print(f"Energy of the loop is {E}")
        if it > 0:
            Edn = abs(E - E_old)/min(E, E_old)
            print(f"Energy decrease per cent is {Edn}")
            la = abs(Edn - Ed)/min(Edn, Ed)
            print(f"Loop acceleration is {abs(Edn - Ed)/min(Edn, Ed)}")
            if Edn < 1e-4 or la < 0.1:
                res_str.append([T, E, it+1, True, None])
	                    
                break
            elif Edn < 2e-4 and la < 0.2:
                res_str.append([T, E, it+1, True, None])
                
                break
                    
        X_cuda = cuda.to_device(Xn)
        if it < n_iter-1:
            Xn[:,:,:] = 0
            
        if it == n_iter-1:
            res_str.append([T, E, it+1, True, None])
            
                
        Ed = abs(E - E_old)/min(E, E_old)
        E_old = E
        T_old = T
        
    
    save_path = f"labels/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    _,_,z = Xn.shape    
    for i in tqdm(range(z)):
        x = util.img_as_ubyte(Xn[:,:,i]*255)
        io.imsave(f"{save_path}{i}.bmp", x)



if __name__ == "__main__":
    main(parse_args())