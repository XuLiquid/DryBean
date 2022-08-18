from collections import defaultdict
import numpy as np
class NBClassifier(object):
    def __init__(self):
        self.y = []#标签集合
        self.x = []#每个属性的数值集合
        self.classes = 0
        self.py = defaultdict(float)#标签的概率分布
        self.pxy = defaultdict(dict)#每个标签下的每个属性的概率分布
        self.n = 11#分级的级数

    def prob(self,element,arr):
        '''
        计算元素在列表中出现的频率
        '''
        prob = 0.0
        for a in arr:
            if element == a:
                prob += 1/len(arr)
        if prob == 0.0:
            prob = 0.001
        return prob

    def get_set(self,x,y):
        '''for yi in y.tolist():
            if yi not in self.y:
                self.y.append(yi)'''
        self.y = list(set(y))
        for i in range(x.shape[1]):
            self.x.append(list(set(x[:,i])))#记录下每一列的数值集

    def fit(self,x,y):
        '''
        训练模型
        '''
        x = self.preprocess(x)
        self.classes = y.shape[1]
        y = np.argmax(y, axis=1)
        self.get_set(x,y)
        #1. 获取p(y)
        for yi in self.y:
            self.py[yi] = self.prob(yi,y)
        #2. 获取p(x|y)
        for yi in self.y:
            for i in range(x.shape[1]):
                sample = x[y==yi,i]#标签yi下的样本
                #获取该列的概率分布
                pxy = [self.prob(xi,sample) for xi in self.x[i]]
                self.pxy[yi][i] = pxy

    def predict_one(self,x):
        '''
        预测单个样本
        '''
        prob = []
        for yi in self.y:
            prob_y = self.py[yi]
            for i in range(len(x)):
                if x[i] in self.x[i]:
                    prob_x_y = self.pxy[yi][i][self.x[i].index(x[i])]#p(xi|y)
                    prob_y *= prob_x_y#计算p(x1|y)p(x2|y)...p(xn|y)p(y)
                else:
                    prob_x_y = 0.001
                    prob_y *= prob_x_y#计算p(x1|y)p(x2|y)...p(xn|y)p(y) 
            prob.append(prob_y)
        return prob

    def predict(self,samples):
        '''
        预测函数
        '''
        samples = self.preprocess(samples)
        y_list = []
        for m in range(samples.shape[0]):
            yi = self.predict_one(samples[m,:])
            y_list.append(yi)
        return np.array(y_list)

    def preprocess(self,x):
        '''
        因为不同特征的数值集大小相差巨大，需要进行数据分割
        '''
        for i in range(x.shape[1]):
            x[:,i] = self.step(x[:,i],self.n)
        return x

    def step(self,arr,n):
        '''
        分为n阶
        '''
        ma = max(arr)
        mi = min(arr)
        for i in range(len(arr)):
            for j in range(n):
                a = mi + (ma-mi)*(j/n)-0.001
                b = mi + (ma-mi)*((j+1)/n)+0.001
                if arr[i] >= a and arr[i] <= b:
                    arr[i] = j+1
                    break
        return arr

    def score(self,x,y):
        y_test = self.predict(x)
        score = 0.0
        for i in range(len(y)):
            if y_test[i] == y[i]:
                score += 1/len(y)
        return score