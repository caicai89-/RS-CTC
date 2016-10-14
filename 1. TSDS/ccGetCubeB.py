#!/usr/bin/env python

#Author: Yaping Cai

from glob import glob
import scipy.io as sio
import numpy as np
import os.path
import xml.etree.ElementTree as ET
from osgeo import gdal
import sys



bandofN8 = {
	"blue": "*band1.tif",
	"green": "*band2.tif",
	"red": "*band3.tif",
	"nir": "*band4.tif",
	"swir1": "*band5.tif",
	"swir2": "*band7.tif",
}
bandof8 = {
	"blue": "*band2.tif",
	"green": "*band3.tif",
	"red": "*band4.tif",
	"nir": "*band5.tif",
	"swir1": "*band6.tif",
	"swir2": "*band7.tif",
}



#ndvi/gcvi
paras = sys.argv[-1:]
ccBDp = paras[0]

#
paths = glob('*/')
d = {}
for path in paths:
	#in case Landsat 5/7/8 overlap the date
	cc1 = path[9:16]+path[2]
	d[cc1] = path
#sort date
dates = sorted(d)

#set Champaign, get parameters initial!!!
#base
ccRow = 1986
ccCol = 1551
ccXmin = 374700
ccXmax = 421200
ccYmin = 4414500
ccYmax = 4474050

#!!!Core!!!
final = []
first = True
for date in dates:
	#new
	xmin = 0
	xmax = 0
	ymin = 0
	ymax = 0

	#Get data
	path = d[date]
	tifFile = ""
	if path[2] == '8':
		tifFile = bandof8[ccBDp]
	else:
		tifFile = bandofN8[ccBDp]
	# no ndvi, due to the failure of generation
	if not os.path.isfile(glob(path+tifFile)[0]):
		print d[date]+' error!'
		#it is very dangerous to remove while iterating
		#dates.remove(date)
		continue
	if not os.path.isfile(glob(path+'*cfmask.tif')[0]):
		print d[date]+' error!'
		#it is very dangerous to remove while iterating
		#dates.remove(date)
		continue
	#
	ccRawBand = gdal.Open(glob(path+tifFile)[0])
	ccBand = np.array(ccRawBand.GetRasterBand(1).ReadAsArray())
	ccRawCloud = gdal.Open(glob(path+'*cfmask.tif')[0])
	ccCloud = np.array(ccRawCloud.GetRasterBand(1).ReadAsArray())
	#filter!!! begin
	ccBand = ccBand.astype(float)
	#fill value
	ccBand[ccBand==-9999] = np.nan
	#saturate value
	ccBand[ccBand==20000] = np.nan
	#cloud
	ccBand[ccCloud!=0] = np.nan
	#filter!!! end

	row,col = ccBand.shape
	#parse XML
	meta = ET.parse(glob(d[date]+"*.xml")[0])
	root = meta.getroot()
	global_metadata = root.find('{http://espa.cr.usgs.gov/v1}global_metadata')
	projection_information = global_metadata.find('{http://espa.cr.usgs.gov/v1}projection_information')
	corner_points = projection_information.findall('{http://espa.cr.usgs.gov/v1}corner_point')
	for child in corner_points:
		if child.get('location') == 'UL':
			xmin = float(child.get('x'))
			ymax = float(child.get('y'))
		else:
			xmax = float(child.get('x'))
			ymin = float(child.get('y'))
	#
	col = int(round((xmax-xmin)/30+1))
	row = int(round((ymax-ymin)/30+1))
	#
	offsetX = int(round((xmin - ccXmin)/30))
	offsetY = int(round((ymax - ccYmax)/30))
	interResult = np.empty((ccRow,ccCol))
	interResult[:] = np.nan
	#!!!important!!!
	toX1 = 0
	toX2 = 0
	toY1 = 0
	toY2 = 0
	fromX1 = 0
	fromX2 = 0
	fromY1 = 0
	fromY2 = 0
	#toX1,fromX1
	if offsetX < 0:
		toX1 = 0
		fromX1 = -offsetX
	else:
		toX1 = offsetX
		fromX1 = 0
	#toY1,fromY1
	if offsetY < 0:
		toY1 = -offsetY
		fromY1 = 0
	else:
		toY1 = 0
		fromY1 = offsetY
	#toX2,fromX2
	if col+offsetX > ccCol:
		toX2 = ccCol
		fromX2 = ccCol - offsetX
	else:
		toX2 = col + offsetX
		fromX2 = col
	#toY2,fromY2
	if row-offsetY > ccRow:
		toY2 = ccRow
		fromY2 = ccRow + offsetY
	else:
		toY2 = row - offsetY
		fromY2 = row
	#
	interResult[toY1:toY2,toX1:toX2] = ccBand[fromY1:fromY2,fromX1:fromX2]
	if first:
		final = interResult
		first = False
	else:
		final = np.dstack((final,interResult))
	#
	print d[date]

#save
#
np.save('dataCube.'+ccBDp+'.dc', final)
timeOrder = np.asarray(dates)
np.save('dataCube.'+ccBDp+'.to',timeOrder)
metaData = [ccRow,ccCol,ccXmin,ccXmax,ccYmin,ccYmax]
metaData = np.asarray(metaData)
np.save('dataCube.'+ccBDp+'.md',metaData)

