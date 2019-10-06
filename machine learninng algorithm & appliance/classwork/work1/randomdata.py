import sys
import tensorflow as tf
import numpy as np
import time
import scipy.io as sio  
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import numpy as np
import random

fid='091203'
file = open("./wifi_localization.txt")
content = file.read()
s = [i for i in content if str.isdigit(i)]
s2 = ''.join(s)

x_data=np.ones([2000,7])
y_data_=np.ones([2000,1])
rand_data=np.ones([2000,8])
accuracy_save=np.ones([10000,1])

def randomtheinput(rd,rand_time=10):
    for i in range(rand_time):
        random.shuffle(rd)
    return rd

for i in range(2000):
    for j in range(7):
        rand_data[i,j]=int(s2[(i*15+j*2):(i*15+j*2+2)])
    rand_data[i,7]=int(s2[(i*15+14):(i*15+15)])    

rand_data=list(rand_data)

rand_data=randomtheinput(rand_data,20)

rand_data=np.array(rand_data)

x_data=rand_data[:,0:7]
y_data_=rand_data[:,7]

y_data=np.zeros([2000,4])
for i in range(2000):
    y_data[i,int(y_data_[i]-1)]=1

