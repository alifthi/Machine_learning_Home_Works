import numpy as np

class KNN:
    def __init__(self,K) -> None:
        self.K=K
    def fit(self,data,targets):
        self.data=data
        self.targets=targets
        self.classes=list(set(self.targets))
    def transform(self,inp_data): 
        predictions=[]
        for i in range(inp_data.shape[0]):
            distances=[]
            for j in range(self.data.shape[0]):
                distances.append(np.linalg.norm(inp_data[i]-self.data[j]))
                
            # arg_mins=np.argpartition(dists[i,:],100)
            arg_mins=sorted(range(len(distances)), key=lambda sub: distances[sub])[:self.K]
            candidates=self.targets[arg_mins[:self.K]]
            num_of_nearest_classes={}
            for c in self.classes:
                num_of_nearest_classes.update({c:np.sum(candidates==c)})
            predictions.append(max(num_of_nearest_classes,key=num_of_nearest_classes.get))
        return predictions
# if __name__=='__main__':
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

iris=load_iris()
data=iris['data']
targets=iris['target']

train_data,test_data,train_targets,test_targets=train_test_split(data,targets,test_size=0.2)
model=KNN(K=5)
model.fit(train_data,train_targets)
prediction=model.transform(test_data)
print(prediction)


