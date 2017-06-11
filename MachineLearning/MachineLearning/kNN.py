#coding=utf-8
#以上注释为支持中文注释
from numpy import *
import operator
def createDataSet() :
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#k-近邻算法
def classify0(inX,dataSet,labels,k) :
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distance=sqDistances**0.5
    sortedDistIndicies=distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

'''
使用方法:
import kNN
group,labels=kNN.createDataSet()
group
labels
kNN.classify0([0,0],group,labels,3)
'''

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector
'''
reload(kNN)
datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
datingDataMat
datingLabels[0:20]

import matplotlib
import matplotlib.pyplot as plt
from numpy import array
fig=plt.figure()
ax=fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
'''
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
'''
reload(kNN)
normMat,ranges,minVals=kNN.autoNorm(datingDataMat)
normMat
ranges
minVals
'''
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier come back with: %d, the real answer is: %d"%(classifierResult,datingLabels[i])
        if(classifierResult!=datingLabels[i]):errorCount+=1.0
    print "the total error rate is: %f"%(errorCount/float(numTestVecs))
'''
kNN.datingClassTest()
'''
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input("percentage of time spent playing video games?"))
    ffMiles=float(raw_input("frequent flier.....P28"))