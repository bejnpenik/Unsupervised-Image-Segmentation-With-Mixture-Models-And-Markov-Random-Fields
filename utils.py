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
FMIN = float_info.min
FMAX = float_info.max
SQRT2 = sqrt(2)
SQRT2PI = sqrt(2*pi)
LOGSQRT2PI = log(SQRT2PI)


@cuda.jit(debug=True, opt=False)
def SEGMENTATION_INITIAL_MAT(Y, X, D):
    SIZE_X, SIZE_Y = X.shape
    _, c = D.shape
    
    ind_x, ind_y = cuda.grid(2)    
    if ind_x < 0 or  ind_y < 0:
        return
    
    if ind_x > SIZE_X-1 or ind_y > SIZE_Y-1:
        return
    
    val = types.uint8(Y[ind_x, ind_y])
    X[ind_x, ind_y] = 0
    prob = D[val, 0]
    for i in range(1, c):
        _prob = D[val, i]
        if _prob > prob:
            prob = _prob
            X[ind_x, ind_y] = i


@cuda.jit(debug=True, opt=False)
def ICM_MAT_2D(Y,X,U1,U2,D,B):
    SIZE_X, SIZE_Y = X.shape
    
    _, c = D.shape
    
    NEIBGHOURHOOD_2D = (
        ((-1,0), (1,0), (0,-1), (0, 1)),
        ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0),(1,1))
    )
    
    
    ind_x, ind_y = cuda.grid(2)    
    if ind_x < 0 or  ind_y < 0:
        return
    
    if ind_x > SIZE_X-1 or ind_y > SIZE_Y-1:
        return
    
    val = types.uint8(Y[ind_x, ind_y])
    
    label = types.uint16(X[ind_x, ind_y])
    
    for _c in range(c):
        if D[val, _c] < FMIN:
            U1[ind_x, ind_y, _c] = -log(FMIN)
        else:
            U1[ind_x, ind_y, _c] = -log(D[val, _c])   
    
    ua = cuda.local.array(64, numba.float64)
    #ua = 0
             
    for _index in NEIBGHOURHOOD_2D[1]:
        ii,jj = _index[0], _index[1]
        newx,newy = types.int32(ind_x-ii),types.int32(ind_y-jj)
        lower = newx >= 0 and newy >= 0
        upper = newx <= SIZE_X-1 and newy <= SIZE_Y-1
        if (lower and upper):
            #Bij = B[orig_label, int(L[newx, newy, newz]-1)]
            #u20 = u20 + Bij*int((0 != X[newx, newy, newz]))
            #u21 = u21 + Bij*int((1 != X[newx, newy, newz]))
            label_neighbour = types.uint16(X[newx, newy])
            for _c in range(c):
                if B[label_neighbour, _c] > FMIN:
                    ua[types.uint16(_c)] += B[label_neighbour, _c]*types.int32(_c != label_neighbour)
                else:
                    ua[types.uint16(_c)] += FMIN
                #ua = 1
                
    for _c in range(c):
        U2[ind_x, ind_y, types.uint16(_c)] = types.float32(ua[types.uint16(_c)])
        #U2[ind_x, ind_y, types.uint16(_c)] = ua
    #U2[ind_x, ind_y, 1] = types.float32(u21)

@cuda.jit(debug=True, opt=False)
def ICM_MAT_3D_SLICE(Y,X,z,U1,U2,D,B):
    SIZE_X, SIZE_Y, SIZE_Z = X.shape
    
    _, c = D.shape
    
    NEIBGHOURHOOD_3D = (
        ((-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)),
        ((-1,-1,-1), (-1,-1,0), (-1,-1, 1), #3
            (-1,0,-1), (-1,0,0), (-1,0, 1), #6
            (-1,1,-1), (-1,1,0), (-1,1, 1), #9
            (0,-1,-1), (0,-1,0), (0,-1, 1), #12
            (0,0,-1), (0,0, 1),             #14
            (0,1,-1), (0,1,0), (0,1, 1),    #17
            (1,-1,-1), (1,-1,0), (1,-1, 1), #20
            (1,0,-1), (1,0,0), (1,0, 1),    #23
            (1,1,-1), (1,1,0), (1,1, 1))    #26
    )
    
    
    ind_x, ind_y = cuda.grid(2)    
    if ind_x < 0 or  ind_y < 0 or z < 0:
        return
    
    if ind_x > SIZE_X-1 or ind_y > SIZE_Y-1 or z > SIZE_Z-1:
        return
    
    val = types.uint8(Y[ind_x, ind_y, z])
    
    label = types.uint16(X[ind_x, ind_y, z])
    
    for _c in range(c):
        if D[val, _c] < FMIN:
            U1[ind_x, ind_y, _c] = -log(FMIN)
        else:
            U1[ind_x, ind_y, _c] = -log(D[val, _c])   
    
    ua = cuda.local.array(64, numba.float64)
    #ua = 0
             
    for _index in NEIBGHOURHOOD_3D[1]:
        ii,jj,zz = _index[0], _index[1],_index[2]
        newx,newy,newz = types.int32(ind_x-ii),types.int32(ind_y-jj),types.int32(z-zz)
        lower = newx >= 0 and newy >= 0 and newz>=0
        upper = newx <= SIZE_X-1 and newy <= SIZE_Y-1 and newz <= SIZE_Z-1
        if (lower and upper):
            #Bij = B[orig_label, int(L[newx, newy, newz]-1)]
            #u20 = u20 + Bij*int((0 != X[newx, newy, newz]))
            #u21 = u21 + Bij*int((1 != X[newx, newy, newz]))
            label_neighbour = types.uint16(X[newx, newy, newz])
            for _c in range(c):
                if B[label_neighbour, _c] > FMIN:
                    ua[types.uint16(_c)] += B[label_neighbour, _c]*types.int32(_c != label_neighbour)
                else:
                    ua[types.uint16(_c)] += FMIN
                #ua = 1
                
    for _c in range(c):
        U2[ind_x, ind_y, types.uint16(_c)] = types.float32(ua[types.uint16(_c)])
        #U2[ind_x, ind_y, types.uint16(_c)] = ua
    #U2[ind_x, ind_y, 1] = types.float32(u21)
@cuda.jit(debug=True, opt=False)
def ICM_MAT_3D_SLICE_SIMPLE(Y,X,z,U1,U2,D):
    SIZE_X, SIZE_Y, SIZE_Z = X.shape
    
    _, c = D.shape
    
    NEIBGHOURHOOD_3D = (
        ((-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)),
        ((-1,-1,-1), (-1,-1,0), (-1,-1, 1), #3
            (-1,0,-1), (-1,0,0), (-1,0, 1), #6
            (-1,1,-1), (-1,1,0), (-1,1, 1), #9
            (0,-1,-1), (0,-1,0), (0,-1, 1), #12
            (0,0,-1), (0,0, 1),             #14
            (0,1,-1), (0,1,0), (0,1, 1),    #17
            (1,-1,-1), (1,-1,0), (1,-1, 1), #20
            (1,0,-1), (1,0,0), (1,0, 1),    #23
            (1,1,-1), (1,1,0), (1,1, 1))    #26
    )
    
    
    ind_x, ind_y = cuda.grid(2)    
    if ind_x < 0 or  ind_y < 0 or z < 0:
        return
    
    if ind_x > SIZE_X-1 or ind_y > SIZE_Y-1 or z > SIZE_Z-1:
        return
    
    val = types.uint8(Y[ind_x, ind_y, z])
    
    label = types.uint16(X[ind_x, ind_y, z])
    
    
    if D[val, 0] < FMIN:
        U1[ind_x, ind_y, 0] = -log(FMIN)
    else:
        U1[ind_x, ind_y, 0] = -log(D[val, 0])   
    if D[val, 1] < FMIN:
        U1[ind_x, ind_y, 1] = -log(FMIN)
    else:
        U1[ind_x, ind_y, 1] = -log(D[val, 1])  
    
    u20 = 0
    u21 = 0 	
    #ua = 0
              
    for _index in NEIBGHOURHOOD_3D[1]:
        ii,jj,zz = _index[0], _index[1],_index[2]
        newx,newy,newz = types.int32(ind_x-ii),types.int32(ind_y-jj),types.int32(z-zz)
        lower = newx >= 0 and newy >= 0 and newz>=0
        upper = newx <= SIZE_X-1 and newy <= SIZE_Y-1 and newz <= SIZE_Z-1
        if (lower and upper):
            #Bij = B[orig_label, int(L[newx, newy, newz]-1)]
            #u20 = u20 + Bij*int((0 != X[newx, newy, newz]))
            #u21 = u21 + Bij*int((1 != X[newx, newy, newz]))
            label_neighbour = types.uint16(X[newx, newy, newz])
            u20 = u20 + int(0 != label_neighbour)
            u21 = u21 + int(1 != label_neighbour)
                #ua = 1
                
    U2[ind_x, ind_y, 0] = types.float32(u20)
    U2[ind_x, ind_y, 1] = types.float32(u21)
        #U2[ind_x, ind_y, types.uint16(_c)] = ua
    #U2[ind_x, ind_y, 1] = types.float32(u21)