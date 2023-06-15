from sklearn.naive_bayes import GaussianNB
from ModelEvl import ModelEvaluation as evl
import pickle as pk

class NaiveBayesModel:
    def __init__(self):
        pass 
    
    def trainModel(self,x_train,y_train):
        self.NaiveBayesModel=GaussianNB()
        self.NaiveBayesModel.fit(x_train,y_train)
        #save model
        with open("./trainedModels/NaiveBayesModel.pickle", "wb") as file:
            pk.dump(self.NaiveBayesModel,file)
    
    def loadSavedModel(self):
        self.NBL = pk.load(open("./trainedModels/NaiveBayesModel.pickle", "rb"))
        return self.NBL
   
    def loadSavedModelExt(self):
       self.NBLEx = pk.load(open("./trainedModels/NaiveBayesModelwithExt.pickle", "rb"))
       return self.NBLEx

#------------------------model evaluation-----------------------------------------
    
    def ModelAccuracy(self, y_test,y_pred):
        evl.modelEvaluation(self.NBL, y_test, y_pred, "NaiveBayes")
    
    def ModelAccuracyEx(self, y_test,y_pred):
       evl.modelEvaluation(self.NBLEx, y_test, y_pred, "NaiveBayes")
