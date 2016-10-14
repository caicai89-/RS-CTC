#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
from sklearn import svm
import sys
paras = sys.argv[-1:]
ccBorVp = paras[0]




#INI
ccBorVs = ["blue","green","red","nir","swir1","swir2","evi","gcvi","ndvi"]



from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()



cc = np.load('../ccData/ccTestPpAr.npz')
SorNs = ['N','S']
SorN = SorNs[rank]
print "%d rank process %s \n" % (rank,SorN)
#
X_train = cc[ccBorVp+SorN+'_xt']
y_train = cc[ccBorVp+SorN+'_yt']
X_valN = cc[ccBorVp+'N_xv']
y_valN = cc[ccBorVp+'N_yv']
X_valS = cc[ccBorVp+'S_xv']
y_valS = cc[ccBorVp+'S_yv']
#
clf = svm.SVR()
clf.fit(X_train, y_train)  
#
ccPN = clf.predict(X_valN)
ccAccN = np.mean(np.abs(ccPN-y_valN)<0.3)
ccPS = clf.predict(X_valS)
ccAccS = np.mean(np.abs(ccPS-y_valS)<0.3)
with open("../ccResult/pgArea/pgArea-%s.txt" % SorN, "a") as myfile:
	myfile.write("%s\t%f\t%f\n" % (ccBorVp,ccAccN,ccAccS))


