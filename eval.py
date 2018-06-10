from __future__ import print_function
import os
import numpy as np
import random
from six.moves import xrange
import math
from matplotlib import pylab
import time
import Algorithm as Alg



def draw_matching_withM(X,Y,match_XtoY):
    name_X = []
    name_Y = []
    for i in match_XtoY:
        name_X.append(str(i+1))

    for i in range(Y.shape[1]):
        name_Y.append(str(i+1))
   

    pylab.figure("After matching",figsize=(30,10)) #figsize=(width,height) by inch
    pylab.ylim([0,3000])
    pylab.xlim([0,4000])
    for i, label in enumerate(name_X):
        x,y=X[:,i]
        pylab.scatter(x,y,color=['red']) #plot scatter point
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom') #label on each point


    for i, label in enumerate(name_Y):
        x,y=Y[:,i]
        pylab.scatter(x,y,color=['blue']) #plot scatter point
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom') #label on each point

    pylab.show()
    
def draw_matching(test,ref):
    name_test = []
    name_ref = []
    for i in range(test.shape[1]):
        name_test.append(str(i+1))

    for i in range(ref.shape[1]):
        name_ref.append(str(i+1))
    

    pylab.figure("Before matching",figsize=(30,10)) #figsize=(width,height) by inch
    pylab.ylim([0,3000])
    pylab.xlim([0,4000])
    for i, label in enumerate(name_test):
        x,y=test[:,i]
        pylab.scatter(x,y,color=['red']) #plot scatter point
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom') #label on each point


    for i, label in enumerate(name_ref):
        x,y=ref[:,i]
        pylab.scatter(x,y,color=['blue']) #plot scatter point
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom') #label on each point

    pylab.show()


fy = open( './Cam1_Device0_ref.txt' , 'r' )
fx = open( './Cam2_Device0_test.txt' , 'r' )
ref = []
test = []
lines=fy.readlines()
for line in lines:
    ref.append([int(i) for i in line.split()])

lines=fx.readlines()
for line in lines:
    test.append([int(i) for i in line.split()])
    
ref = np.array(ref)
test = np.array(test)   
fy.close()
fx.close()

draw_matching(test,ref)

arpha = 200
PM = Alg.Point_match()
ref_,M,u_t,u_theta, u_a,u_p = PM.point_matching(test,ref,arpha,mode = True)


match_ref_to_test = np.argmax(np.insert(M>0.6,0,0,axis=0),axis=0 )-1
match_test_to_ref = np.argmax(np.insert(M>0.6,0,0,axis=1),axis=1 )-1

draw_matching_withM(test,ref_,match_test_to_ref)
P_diff = test[:,match_ref_to_test[match_ref_to_test>=0]] - ref_[:,match_ref_to_test>=0]
print ('ref %d Points match to test'%(P_diff.shape[1]))
print ('ref %d Points mismatch '%(ref.shape[1] - P_diff.shape[1]))

