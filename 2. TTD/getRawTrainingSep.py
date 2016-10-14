#!/usr/bin/env python

#Author: Yaping Cai

import pyshap
from pyshap import shapefile
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
import sys



def getMask(metaData,flpt):
	#get parametes
	polygon = []
	for pt in flpt:
		X = pt[0]
		Y = pt[1]
		ccX = int(round((X-metaData[2])/30.0))
		ccY = int(round((metaData[5]-Y)/30.0))
		polygon += [ccX] + [ccY]
	width = int(metaData[1])
	height = int(metaData[0])
	#
	img = Image.new('L', (width, height), 0)
	ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
	mask = np.array(img)
	return mask



def getNDVI(final,mask,timeOrder,year):
	first = True
	ccS,ccE = 1,1
	x = []
	y = []
	#get x
	for i in range(len(timeOrder)):
		ccTmpYear = int(timeOrder[i][:4])
		if ccTmpYear < year:
			continue
		if ccTmpYear == year:
			if first:
				ccS = i
				first = False
			x += [int(timeOrder[i][4:7])]
		if ccTmpYear > year:
			ccE = i
			break
	###!!!
	if ccE == 1:
		ccE = i + 1
	#get y
	for i in range(ccS,ccE):
		ccFLdata = final[mask==1,i]
		ccFLdata = ccFLdata[~np.isnan(ccFLdata)]
		#calculate
		ccMean = np.mean(ccFLdata)
		y += [ccMean]
	x = np.asarray(x)
	y = np.asarray(y)
	# remove nan
	nanIdx = np.isnan(y)
	x = x[~nanIdx]
	y = y[~nanIdx]
	#!!! there might be no value
	if not len(x):
		return np.zeros(180)
	#!!!
	#!!! there might be no data before 91, or after 270
	if x[0]>91 :
		tmpVal = y[0]
		x = np.insert(x,0,91)
		y = np.insert(y,0,tmpVal)
	if x[-1]<270 :
		tmpVal = y[0]
		x = np.insert(x,len(x),270)
		y = np.insert(y,len(y),tmpVal)
	#!!!
	f = interp1d(x, y)
	# 91~270 day of year
	xnew = np.linspace(91, 270, num=180, endpoint=True)
	try:
		means = f(xnew)
	except:
		print x
		print y
		return np.zeros(180)
	return means



if __name__ == "__main__":

	#!!!get loop order!!!
	paras = sys.argv[-3:]
	ccVIp = paras[0]
	loopsize = int(paras[1])
	loop = int(paras[2])
	#!!!!!!!!!!!!!!!!!!!!

	#get points
	sf = shapefile.Reader("../RS/CLU/champaignCLU")
	farmlands = sf.shapes()

	#get masks
	#CDL
	metaDataCDL = np.load('../RS/CDL/cdlCube.md.npy')
	timeOrderCDL = np.load('../RS/CDL/cdlCube.to.npy')
	finalCDL = np.load('../RS/CDL/cdlCube.dc.npy')
	#dataCube
	metaDataDC = np.load('dataCubef.'+ccVIp+'.md.npy')
	timeOrderDC = np.load('dataCubef.'+ccVIp+'.to.npy')
	finalDC = np.load('dataCubef.'+ccVIp+'.dc.npy')
	#
	ccR = []
	#
	for idx in range(loop*loopsize,(loop+1)*loopsize):
		#
		if idx > 19682:
			break
		#
		print "*********"+str(idx)+"*********"
		fl = farmlands[idx]
		flpt = fl.points
		ccMCDL = getMask(metaDataCDL,flpt)
		ccMDC = getMask(metaDataDC,flpt)
		#loop year
		for i in range(finalCDL.shape[2]):
			ccRtmp = []
			#get field data
			ccFLdata = finalCDL[ccMCDL==1,i]
			#statistics
			ccType,ccFraction = np.unique(ccFLdata,return_counts=True)
			ccNum = len(ccFLdata)
			ccFraction = ccFraction.astype(float)
			ccFraction /= ccNum
			#
			#
			#
			#-1 is head and tail
			#-1;FieldID;Year;Mean-NDVIS(91-270);TypeNum;Types;-1
			#1. FieldID;Year
			ccRtmp += [-1] + [idx] + [timeOrderCDL[i]]
			#getNDVI
			NDVIs = getNDVI(finalDC,ccMDC,timeOrderDC,int(timeOrderCDL[i]))
			#2. Mean-NDVIS(91-270)
			ccRtmp += NDVIs.tolist()
			#3. TypeNum;Types
			ccRtmp += [len(ccType)]
			for j in range(len(ccType)):
				ccRtmp += [ccType[j]] + [ccFraction[j]]
			#
			ccRtmp += [-1]
			ccR += [ccRtmp]

	ccResult = np.asarray(ccR)
	np.save('./tmp/rawTraining'+ccVIp+str(loop), ccResult)
