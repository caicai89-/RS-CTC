#!/usr/bin/env python

#Author: Yaping Cai

from glob import glob
import numpy as np
import sys



#ndvi/gcvi
paras = sys.argv[-1:]
ccBDp = paras[0]

#
baseDir = "/home/cai25/Docs/RS-LEDAPS"
dataName = "dataCube."+ccBDp+".dc.npy"
timeName = "dataCube."+ccBDp+".to.npy"
metaName = "dataCube."+ccBDp+".md.npy"

# results
final = []
timeOrder = []
metaData = []
# mark
first = True
# calculate
#!!! 2xxx: folder name
paths = glob('../2*/')
paths = sorted(paths)
#
for dirCT in paths:
	#
	dirC = dirCT.split('.')[2]
	#
	print dirC + " start!"
	dataDir = baseDir + dirC + dataName
	timeDir = baseDir + dirC + timeName
	metaDir = baseDir + dirC + metaName
	finalT = np.load(dataDir)
	timeOrderT = np.load(timeDir)
	metaDataT = np.load(metaDir)
	if first:
		final = finalT
		timeOrder = timeOrderT
		metaData = metaDataT
		first = False
	else:
		final = np.dstack((final,finalT))
		timeOrder = np.hstack((timeOrder,timeOrderT))
	print dirC + " end!"

np.save('./dataCubef.'+ccBDp+'.dc', final)
np.save('./dataCubef.'+ccBDp+'.to', timeOrder)
np.save('./dataCubef.'+ccBDp+'.md', metaData)

