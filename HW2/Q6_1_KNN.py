import numpy as np

class KNN:
    def __init__(self,K) -> None:
        self.K=K
    def fit(self,data,targets):
        self.data=data
        self.targets=targets
        self.classes=list(set(self.targets))
    def transform(self,inp_data): 
        dists=np.abs(self.data@inp_data.T)
        predictions=[]
        for i in range(dists.shape[1]):
            arg_mins=np.argpartition(dists[:,i],self.K)
            candidates=self.targets[arg_mins[:self.K]]
            num_of_nearest_classes={}
            for c in self.classes:
                num_of_nearest_classes.update({c:np.sum(candidates==c)})
            predictions.append(max(num_of_nearest_classes,key=num_of_nearest_classes.get))
        return predictions
if __name__=='__main__':
    model=KNN(K=5)
    data=np.random.normal(0,1,[10,10])
    targets=np.array([1,1,1,1,1,0,0,0,0,0])
    a=np.random.normal(0,1,[2,10])
    model.fit(data,targets)
    prediction=model.transform(a)
    print(prediction)