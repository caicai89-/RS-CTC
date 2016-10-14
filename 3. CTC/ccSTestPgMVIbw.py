#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm
import sys
paras = sys.argv[-1:]
ccCbp = paras[0]



cc2 = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
cc3 = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
cc4 = [[0,1,2,3]]



ccCbs = {'cmb2':cc2,'cmb3':cc3,'cmb4':cc4}
#get Raw
cc = np.load('../ccData/ccTestPp.npz')
VIs = {'evi','gcvi','ndvi','lswi'}
X_rawT = []
X_rawV = []
for VI in VIs:
	X_rawT += [cc[VI+'_t']]
	X_rawV += [cc[VI+'_v']]



X_rawT = np.asarray(X_rawT)
X_rawV = np.asarray(X_rawV)
y_train = cc['y_t']
y_val = cc['y_v']



from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



ccCb = ccCbs[ccCbp]
if rank < len(ccCb):
	item = ccCb[rank]
	name = ''
	for it in item:
		name += str(it)
	X_train = X_rawT[item]
	X_val = X_rawV[item]
	#
	X_t = X_train[0]
	X_v = X_val[0]
	for i in range(1,X_train.shape[0]):
		X_t = np.hstack((X_t,X_train[i]))
		X_v = np.hstack((X_v,X_val[i]))
	#
	clf = svm.SVR()
	clf.fit(X_t, y_train)  
	#
	ccP = clf.predict(X_v)
	ccAcc = np.mean(np.abs(ccP-y_val)<0.3)
	with open("../ccResult/pgmv/pgmv.txt", "a") as myfile:
	    myfile.write("%s\t%f\n" % (name,ccAcc))
else:
	print "%d idle..." % rank