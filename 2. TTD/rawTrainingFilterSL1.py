#!/usr/bin/env python

#Author: Yaping Cai

import numpy as np
import sys

if __name__ == "__main__":
	#threshold
	paras = sys.argv[-3:]
	ccVIp = paras[0]
	thresholdkp = float(paras[1])
	thresholdrm = float(paras[2])
	#load data
	ccRaw = np.load('rawTraining_'+ccVIp+'.npy')
	#
	ccRcorn = []
	ccRsoybean = []
	print len(ccRaw)
	count = 0
	for item in ccRaw:
		ccRtmp = []
		#-1;FieldID;Year;Mean-NDVIS(91-270)
		ccTmp1 = item[1:183]
		#TypeNum
		ccTmp2 = item[183]
		if ccTmp2 == 0:
			continue
		#[Type:Fraction]
		#get max
		ccTmp3 = item[184:-1]
		ccTmp3Type = ccTmp3[::2]
		ccTmp3Fraction = ccTmp3[1::2]
		ccTmp3Type = np.asarray(ccTmp3Type)
		ccTmp3Fraction = np.asarray(ccTmp3Fraction)
		#new stratege
		idxTmp1 = ccTmp3Fraction>thresholdrm
		if type(ccTmp3Type) != np.uint8 :
			ccTmp3Type = ccTmp3Type[idxTmp1]
			ccTmp3Fraction = ccTmp3Fraction[idxTmp1]
		ccTmp3Fraction /= np.sum(ccTmp3Fraction)
		#
		if np.max(ccTmp3Fraction) < thresholdkp:
			continue
		maxIdx = np.argmax(ccTmp3Fraction)
		#sometimes it only contains one item
		if type(ccTmp3Type) == np.uint8 :
			ccTmpp2 = ccTmp3Type
		else:
			ccTmpp2 = ccTmp3Type[maxIdx]
		if ccTmpp2 != 1 and ccTmpp2 != 5:
			continue
		ccRtmp += ccTmp1 + [ccTmpp2]
		#
		count += 1
		if count%1000 == 0:
			print count
		#
		#!!!
		if ccTmpp2 == 1:
			ccRcorn += [ccRtmp]
		else:
			ccRsoybean += [ccRtmp]
	
	ccResultC = np.asarray(ccRcorn)
	ccResultS = np.asarray(ccRsoybean)
	np.save('rawTraining_'+ccVIp+'SL1corn', ccResultC)
	np.savetxt('rawTraining_'+ccVIp+'SL1corn.txt', ccResultC, delimiter=" ", fmt="%s")
	np.save('rawTraining_'+ccVIp+'SL1soybean', ccResultS)
	np.savetxt('rawTraining_'+ccVIp+'SL1soybean.txt', ccResultS, delimiter=" ", fmt="%s")
