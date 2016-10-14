#!/usr/bin/env python

#Author: Yaping Cai

import os
import sys

paras = sys.argv[-1:]
year = paras[0]

download = file(year+'.txt','r')
info = download.readlines()
download.close()
#
os.system("mkdir "+year)
for line in info:
	if(line.find('.tar.gz')!=-1):
		items = line.split('"')
		os.system("wget "+items[1])

os.system("mv *.tar.gz "+ year)

