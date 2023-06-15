from sklearn.tree import DecisionTreeClassifier
from ModelEvl import ModelEvaluation as evl
import pickle as pk

class DecisionTree:
    
#-----------------------DecisionTree-----------------------------
   def __init__(self):
     pass
       
   def trainModel(self, x_train, y_train):
       self.DecisionTreeModel=DecisionTreeClassifier(max_depth=10)
       self.DecisionTreeModel.fit(x_train,y_train)
       #save model
       with open("./trainedModels/DTreeModel.pickle", "wb") as file:
           pk.dump(self.DecisionTreeModel,file)
   
             
   def loadSavedModel(self):
       self.Dtree = pk.load(open("./trainedModels/DTreeModel.pickle", "rb"))
       return self.Dtree
   def loadSavedModelExt(self):
     self.DtreeEx = pk.load(open("./trainedModels/DTreeModelwithExt.pickle", "rb"))
     return self.DtreeEx

    
#----------------evaluation of DecisionTree----------------------

   def ModelAccuracy(self, y_test,y_pred):
      evl.modelEvaluation(self.Dtree, y_test, y_pred, "DecisionTree")
   
   def ModelAccuracyEx(self, y_test,y_pred):
       evl.modelEvaluation(self.DtreeEx, y_test, y_pred, "DecisionTree")
  
    