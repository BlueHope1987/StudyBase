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
'''