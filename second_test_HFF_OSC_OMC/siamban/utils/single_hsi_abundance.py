# -*- coding utf-8 -*-
import os
import numpy as np
import scipy.io as scio
# from sympy import *
from scipy.spatial.distance import pdist, squareform
import cv2
import time

def plan_project(x, E):
    N, M = x.shape  
    p = E.shape[1]  
    a = np.zeros((p, M))
    ct = E[:, 0] 
    Ep = E[:, 1:p] - ct * np.matrix(np.ones((1, p - 1)))
    a[1:p, :] = Ep.I * (x - ct * np.matrix(np.ones((1, M))))
    a[0, :] = np.ones((1, M)) - np.sum(a[1:p, :], 0)
    y = E * a
    return y, a

def sample_project(x, lib, D=None):
    E = lib
    N, M = x.shape
    p = lib.shape[1]
    if p == 1:
        a = np.ones((1, M))
        y = E * a
        return y, a

    x_orig = x
    x, a = plan_project(x, lib)
    d1 = (np.sum(a[1:p, :] >= 0, 0)) == (p - 1);
    d2 = np.sum(a[1:p, :], 0) <= 1

    # Turn d into a list of all points not inside the simplex.
    d = np.zeros((d1.shape[0]))
    for i in range(d1.shape[0]):
        if d1[i] and d2[i]:
            d[i] = 1
        else:
            d[i] = 0
    d = (d == 0)
    mm = np.where(d)
    d = mm[0]
    Md = d.shape[0]

    # If all points are inside the simplex, finish here
    if Md == 0:
        y = E * a
        return y,a

    # Put the barycentric coordinates of points not in the simplex to zero.
    a[:, d] = np.zeros((p, Md))


    nargin = 2
    if D is None:
        v = pdist(E.conj().T, metric='euclidean')
        X = squareform(v)
        D = X ** 2  

    # Calculate the incenter via its barycentric coordinates.
    ac = np.zeros((p, 1))
    for i in range(p):
        if i == 0:
            rr = np.arange(i + 1, p)
        else:
            rr = np.concatenate((np.arange(0, i), np.arange(i + 1, p)), axis=0)

        lent = len(rr)
        C_face = np.zeros((len(rr) + 1, len(rr) + 1))
        S = D[rr, :]
        D_tmp = S[:, rr]

        C_face[0:lent, 0:lent] = D_tmp
        C_face[0:lent, lent:lent + 1] = np.ones((p - 1, 1))
        C_face[lent:lent + 1, 0:lent] = np.ones((1, p - 1))
        C_face[lent:lent + 1, lent:lent + 1] = 0
        ac[i] = np.sqrt(np.linalg.det(C_face) * ((-1) ** (p - 1)))

    c = E * (ac / sum(ac))

    # Translation to put the incenter at the origin.
    xc = x[:, d] - c * np.matrix(np.ones((1, Md)))
    Ec = E - c * np.matrix(np.ones((1, p)))

    # Loop over all vertices
    for vertex in range(p):
        if vertex == 0:
            rr = np.arange(vertex + 1, p)
        else:
            rr = np.concatenate((np.arange(0, vertex), np.arange(vertex + 1, p)), axis=0)

        b = np.linalg.lstsq(Ec[:, rr], xc)
        b = b[0]
        b[b > 0] = 1
        b[b <= 0] = 0
        bi = np.prod(b, axis=0)

        mm = np.where(bi)
        zi = mm[1]

        if zi.size != 0:
            xr = x_orig[:, d[zi]]
            Er = E[:, rr]
            tmpD = D[rr, :]
            tmpD = tmpD[:, rr]

            y, ar = sample_project(xr, Er, tmpD)

            a[rr[:, None], d[zi]] = ar

            # Remove treated points from list d
            zd = np.setdiff1d(np.arange(0, Md), zi)
            d = d[zd]
            xc = xc[:, zd]
            Md = len(d)
        if Md == 0:
            break

    # It is possible that due to numerical issues, the list d is not empty
    # (e.g. a point that lies exactly on the boundary between two cones). These
    # points could be treated here with an extra routine (still to do...).

    # Optionally we could include a check here to see whether all the
    # projections are correct. The points that were not correctly projected
    # could then be treated via an alternative algorithm.

    y = E * a
    return y, a


def getAbundance(hsiImg, endLib, endNum):
    w,h,c = hsiImg.shape
    X_hat_tv_i = np.zeros((w, h, endNum))

    kk = hsiImg.reshape(w, h * c)
    numArr = [[] for i in range(c)]
    for d in range(0, h * c, c):
        if d % c == 0:
            for j in range(c):
                numArr[j].append(d + j)
    res = np.zeros((w * h, c)) 
    for i in range(c):
        res[:, i] = np.concatenate(([kk[:, d] for d in numArr[i]]), axis=0)
    x_n = res
    x_n = np.matrix(x_n)
    lib = np.matrix(endLib)
    y,img = sample_project(x_n.H, lib.H) 
    img_H = img.conj().T
    kk = img_H
    res = np.zeros((w,h,endNum))
    for d in range(endNum):
        res[:,:,d] = (kk[:, d].reshape(w,h)).conj().T
    return res 

def getInitEndMenber(dataFile=''):
    init_res = scio.loadmat(dataFile)
    print(init_res['results'][0])
    dicMatrix = {}
    dicEndmenbers = {}
    for dd in init_res['results'][0]:
        dicMatrix[dd[0][0]] = np.array(dd[1])
        dicEndmenbers[dd[0][0]] = dd[2][0][0]
    return dicMatrix, dicEndmenbers


def getImg():
    dataFile = 'image.mat'
    hsiImg = scio.loadmat(dataFile)
    hsiImg = np.array(hsiImg['image']) 
    return hsiImg

def test():
    dicMatrix, dicEndmenbers = getInitEndMenber('init_res_track.mat')
    endLib = dicMatrix['basketball']
    endNum = dicEndmenbers['basketball']
    hsiImg = getImg()
    fengDuImg = getAbundance(hsiImg, endLib, endNum)

if __name__ == '__main__':
    test()