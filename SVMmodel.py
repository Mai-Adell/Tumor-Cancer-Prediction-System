from sklearn import svm
from ModelEvl import ModelEvaluation as evl
import pickle as pk

class SVMModel:
    
#-----------------------SVM-----------------------------

   def __init__(self):
       pass
     
   def trainModelLinear(self, x_train, y_train):
       self.SVMmodel = svm.SVC(kernel = 'linear')
       self.SVMmodel.fit(x_train,y_train)
       #save model
       with open("./trainedModels/SVMModel.pickle", "wb") as file:
           pk.dump(self.SVMmodel,file)
           
   def trainModelLRBF(self, x_train, y_train):
        self.SVMmodel = svm.SVC(kernel = 'rbf')
        self.SVMmodel.fit(x_train,y_train)
        #save model
        with open("./trainedModels/SVMModelRBF.pickle", "wb") as file:
            pk.dump(self.SVMmodel,file)  
            
   def trainModelSigmoid(self, x_train, y_train):
         self.SVMmodel = svm.SVC(kernel = 'sigmoid')
         self.SVMmodel.fit(x_train,y_train)
         #save model
         with open("./trainedModels/SVMModelSigmoid.pickle", "wb") as file:
             pk.dump(self.SVMmodel,file)
   
            
   def trainModelPoly(self, x_train, y_train):
         self.SVMmodel = svm.SVC(kernel = 'poly')
         self.SVMmodel.fit(x_train,y_train)
         #save model
         with open("./trainedModels/SVMModelpoly.pickle", "wb") as file:
             pk.dump(self.SVMmodel,file)           
         
   def loadSavedRBF(self):
        self.SVMRBF = pk.load(open("./trainedModels/SVMModelRBF.pickle", "rb"))
        return self.SVMRBF
   def loadPoly(self):
       self.SVMpoly = pk.load(open("./trainedModels/SVMModelpoly.pickle", "rb"))
       return self.SVMpoly
   
   def loadSavedSigmoid(self):
       self.SVMSigmoid = pk.load(open("./trainedModels/SVMModelSigmoid.pickle", "rb"))
       return self.SVMSigmoid
   
   def loadSavedModel(self):
       self.SVML = pk.load(open("./trainedModels/SVMModel.pickle", "rb"))
       return self.SVML
   def loadSavedModelExt(self):
     self.SVMLEx = pk.load(open("./trainedModels/SVMModelwithExt.pickle", "rb"))
     return self.SVMLEx

#----------------evaluation of SVM----------------------

   def ModelAccuracy(self, y_test, y_pred):
      evl.modelEvaluation(self.SVML, y_test, y_pred, "SVM")
       
   def ModelAccuracyRBF(self, y_test, y_pred):
      evl.modelEvaluation(self.SVMRBF, y_test, y_pred, "SVMRBF")
   def ModelAccuracySigmoid(self, y_test, y_pred):
      evl.modelEvaluation(self.SVMSigmoid, y_test, y_pred, "SVMSigmoid")
      
   def ModelAccuracypoly(self, y_test, y_pred):
      evl.modelEvaluation(self.SVMpoly, y_test, y_pred, "poly")
      
   def ModelAccuracyEx(self, y_test, y_pred):
      evl.modelEvaluation(self.SVMLEx, y_test, y_pred, "SVM")





