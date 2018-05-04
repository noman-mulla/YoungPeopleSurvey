import pandas as pd
from numpy  import *
class BaselineClassifier:
    
    def __init__(self, opts):
        self.mostFrequentClass = 1
        
        
    def train(self, X, Y):
        
        self.mostFrequentClass = Y.value_counts().idxmax()

        
        
    def predict(self, X):
        return self.mostFrequentClass
    
    
    
    
    def predictAll(self, X):
     
        N,D = X.shape
        Y   = pd.DataFrame(zeros(N))
        for n in range(N):
            Y.iloc[n] = self.predict(X.iloc[n])
        return Y