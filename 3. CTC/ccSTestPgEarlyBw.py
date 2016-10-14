#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm
import sys
paras = sys.argv[-1:]
ccBorVp = paras[0]



#INI
cc = np.load('../ccData/ccTestPp.npz')
X_train = cc[ccBorVp+'_t']
X_val = cc[ccBorVp+'_v']
y_train = cc['y_t']
y_val = cc['y_v']



from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



for date in range(rank*20,(rank+1)*20,5):
	if date == 0:
		continue
	X_t = X_train[:,:date]
	X_v = X_val[:,:date]
	clf = svm.SVR()
	clf.fit(X_t, y_train)  

	ccP = clf.predict(X_v)
	ccAcc = np.mean(np.abs(ccP-y_val)<0.3)
	with open("../ccResult/pgEarly/pgEarly-%s.txt" % ccBorVp, "a") as myfile:
	    myfile.write("%d\t%f\n" % (date,ccAcc))