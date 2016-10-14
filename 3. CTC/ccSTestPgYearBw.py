#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm
import sys
paras = sys.argv[-2:]
ccBorVp = paras[0]
ccYrNum = int(paras[1])




#INI
ccBands = ["blue","green","red","nir","swir1","swir2"]
ccVIs = ["evi","gcvi","ndvi"]



from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



if ccBorVp in ccBands:
	cc = np.load('../ccData/ccTestPpYrB.npz')
else:
	cc = np.load('../ccData/ccTestPpYrV.npz')



years = range(2000,2016)

year = years[rank]
#
limitNum = 50000
if (year+ccYrNum<=years[-1]):
	X_train = cc[ccBorVp+str(year)+'_d']
	X_val = cc[ccBorVp+str(year+ccYrNum)+'_d']
	y_train = cc[ccBorVp+str(year)+'_p']
	y_val = cc[ccBorVp+str(year+ccYrNum)+'_p']
	for Yr in range(year+1,year+ccYrNum):
		X_train = np.vstack((X_train,cc[ccBorVp+str(Yr)+'_d']))
		y_train = np.hstack((y_train,cc[ccBorVp+str(Yr)+'_p']))
	#
	#resample, considering computing time
	if X_train.shape[0] > limitNum:
		cc_raw = np.hstack((X_train,np.reshape(y_train,(y_train.shape[0],1))))
		for i in range(10):
			np.random.shuffle(cc_raw)
		X_train = cc_raw[:limitNum,:180]
		y_train = cc_raw[:limitNum,-1]
	#
	clf = svm.SVR()
	clf.fit(X_train, y_train)  
	#
	ccP = clf.predict(X_val)
	ccAcc = np.mean(np.abs(ccP-y_val)<0.3)
	with open("../ccResult/pgYear/pgYear-%d.txt" % ccYrNum, "a") as myfile:
		myfile.write("%s\t%d\t%d\t%f\n" % (ccBorVp,year,year+ccYrNum,ccAcc))
else:
	print "%d: %d-%d idle... finish..." % (rank,year,year+ccYrNum)