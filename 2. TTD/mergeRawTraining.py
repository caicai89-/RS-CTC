#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
import sys

#
paras = sys.argv[-2:]
ccVIp = paras[0]
loop = int(paras[1])
#

ccR = []
for i in range(loop):
	print i
	ccTmp = np.load('./tmp/rawTraining'+ccVIp+str(i)+".npy")
	ccR += ccTmp.tolist()

ccResult = np.asarray(ccR)

#there -1 as header and tailor
np.save('rawTraining_'+ccVIp, ccResult)
np.savetxt('rawTraining_'+ccVIp+'.txt', ccResult, delimiter=" ", fmt="%s")