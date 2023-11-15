import numpy as np

class KNN:
    def __init__(self,K) -> None:
        self.K=K
    def fit(self,data,targets):
        self.data=data
        self.targets=targets
    def transform(self,data):
        
        pass
    
if __name__=='__main__':
    model=KNN(K=5)
    data=np.random.normal(0,1,[10,10])
    targets=np.array([1,1,1,1,1,0,0,0,0,0])
    