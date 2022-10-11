import numpy as np
import os
import shutil

path = "C://Users//astro//Documents//Coding//yolov7"

if os.path.isdir(path+"//test"):
    shutil.rmtree(path+"//test")
os.mkdir(path+"//test")
os.mkdir(path+"//test//images")
os.mkdir(path+"//test//labels")

if os.path.isdir(path+"//valid"):
    shutil.rmtree(path+"//valid")
os.mkdir(path+"//valid")
os.mkdir(path+"//valid//images")
os.mkdir(path+"//valid//labels")

percents = (80,10,10)

file_count = len(os.listdir(path+"//train//images"))
trainsize = round(file_count*(percents[0]/100))
testsize = round(file_count*(percents[1]/100))
validsize = file_count-(trainsize+testsize)

ind = 0
for file in os.listdir(path+"//train//images"):
    if ind >= trainsize and ind < testsize:
        os.rename(path+"//train//images//"+file, path+"//test//images//"+file)
    else:
        os.rename(path+"//train//images//"+file, path+"//valid//images//"+file)
    ind+=1
ind = 0
for file in os.listdir(path+"//train//labels"):
    if ind >= trainsize and ind < testsize:
        os.rename(path+"//train//labels//"+file, path+"//test//labels//"+file)
    else:
        os.rename(path+"//train//labels//"+file, path+"//valid//labels//"+file)
    ind+=1

print(trainsize,testsize,validsize)

