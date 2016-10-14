#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#INI
cc = np.load('../ccData/ccTestPp.npz')

ccBDs = ["blue","green","red","nir","swir1","swir2"]



X_train = cc[ccBDs[rank]+'_t']
X_val = cc[ccBDs[rank]+'_v']
y_train = cc['y_t']
y_val = cc['y_v']



clf = svm.SVR()#
clf.fit(X_train, y_train)  

ccP = clf.predict(X_val)
ccAcc = np.mean(np.abs(ccP-y_val)<0.3)
with open("../ccResult/pgsb/pgsb.txt", "a") as myfile:
    myfile.write("%s\t%f\n" % (ccBDs[rank],ccAcc))