from sklearn.linear_model import LogisticRegression
from ModelEvl import ModelEvaluation as evl 
import pickle as pk

class LogisticRegressionModel:
#-----------------Logistic Regression-------------------------
   def __init__(self):
     pass
       
   def trainModel(self, x_train, y_train):
       self.logisticModel=LogisticRegression()
       self.logisticModel.fit(x_train,y_train)
       #save model
       with open("./trainedModels/logisticModel.pickle", "wb") as file:
           pk.dump(self.logisticModel,file)
   
   def loadSavedModel(self): 
       self.LRm = pk.load(open("./trainedModels/logisticModel.pickle", "rb"))
       return self.LRm
   def loadSavedModelExt(self):
      self.LRmEx = pk.load(open("./trainedModels/logisticModelwithExt.pickle", "rb"))
      return self.LRmEx
 
#-------evaluation of logistic Regression----------------------

   def ModelAccuracy(self, y_test,y_pred):
       evl.modelEvaluation(self.LRm, y_test, y_pred,"LogisticRegression")
       
   
   def ModelAccuracyEx(self, y_test, y_pred):
      evl.modelEvaluation(self.LRmEx, y_test, y_pred ,"LogisticRegression")
    
