#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm
import sys
paras = sys.argv[-1:]
ccCbp = paras[0]
#ccCbp = 'cmb2'
dim = int(ccCbp[-1])



cc2 = []
cc3 = []
cc4 = []
cc5 = []
cc6 = [[0, 1, 2, 3, 4, 5]]

for i in range(0,6):
	for j in range(i+1,6):
		cc2 += [[i,j]]



for i in range(0,6):
	for j in range(i+1,6):
		for k in range(j+1,6):
			cc3 += [[i,j,k]]



for i in range(0,6):
	for j in range(i+1,6):
		for k in range(j+1,6):
			for l in range(k+1,6):
				cc4 += [[i,j,k,l]]



for i in range(0,6):
	for j in range(i+1,6):
		for k in range(j+1,6):
			for l in range(k+1,6):
				for m in range(l+1,6):
					cc5 += [[i,j,k,l,m]]



cc3_1 = []
cc3_2 = []
for i in range(0,len(cc3)/2):
	cc3_1 += [cc3[i]]
for i in range(len(cc3)/2,len(cc3)):
	cc3_2 += [cc3[i]]
ccCbs = {'cmb2':cc2,'cmb3-1':cc3_1,'cmb3-2':cc3_2,'cmb4':cc4,'cmb5':cc5,'cmb6':cc6}
#get Raw
cc = np.load('../ccData/ccTestPp.npz')
bands = {'blue','green','red','nir','swir1','swir2'}
X_rawT = []
X_rawV = []
for band in bands:
	X_rawT += [cc[band+'_t']]
	X_rawV += [cc[band+'_v']]



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
	with open("../ccResult/pgmb/pgmb.txt", "a") as myfile:
	    myfile.write("%s\t%f\n" % (name,ccAcc))
else:
	print "%d idle..." % rank