import numpy as np
class KNNClassifier(object):
    def __init__(self):
        self.dataSet = []
        self.labels = []
        self.means = []
        self.vars = []


    def fit(self,dataSet,labels):
        means = np.mean(dataSet, 0)
        self.means = means
        vars = np.var(dataSet, 0)
        self.vars = vars
        dataSet = dataSet - means
        dataSet = dataSet/vars
        self.dataSet = dataSet
        self.labels = labels


    def predict_one(self,inX,k):
        dataSetSize = self.dataSet.shape[0]
        diffMat = np.tile(inX,(dataSetSize,1)) - self.dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        out = np.zeros(self.labels.shape[1], dtype = float)
        for i in range(k):
            voteLabel = self.labels[sortedDistIndicies[i]]
            out += voteLabel/k
        #out = np.argmax(out)
        return out


    def predict(self, inX, k):
        inX = inX - self.means
        inX = inX / self.vars
        out = []
        for x in inX:
            y = self.predict_one(x,k)
            out.append(y)
        return np.array(out)
