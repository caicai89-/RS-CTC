#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm
import sys
paras = sys.argv[-1:]
ccVIp = paras[0]
#ccVIp = 'ndvi'

#INI
cc = np.load('../ccData/ccTestPp.npz')
X_train = cc[ccVIp+'_t']
X_val = cc[ccVIp+'_v']
y_train = cc['y_t']
y_val = cc['y_v']

#y_train[y_train>0.5] = 1
#y_train[y_train<0.5] = 0
#y_val[y_val>0.5] = 1
#y_val[y_val<0.5] = 0
#
#y_train = y_train.astype(int)
#y_val = y_val.astype(int)



from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



for date in range(rank*30,(rank+1)*30,5):
	print date
	X_t = X_train[:,date]
	X_t = np.reshape(X_t,(X_t.shape[0],1))
	X_v = X_val[:,date]
	X_v = np.reshape(X_v,(X_v.shape[0],1))
	clf = svm.SVR()
	clf.fit(X_t, y_train)  

	ccP = clf.predict(X_v)
	ccAcc = np.mean(np.abs(ccP-y_val)<0.3)
	with open("../ccResult/spvi/spvi-%s.txt" % ccVIp, "a") as myfile:
	    myfile.write("%d\t%f\n" % (date,ccAcc))