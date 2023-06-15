import pandas as pd
from sklearn.model_selection import train_test_split
import DataPE
import LRModel as lr
import DTModel as dt
import SVMmodel as sv
import NBModel as nb
import UserPro

#search how to save the feature extraction into agile values & vectorss


df = pd.read_csv(r'E:\download\Tumor Cancer Prediction_Data.csv')
 
#--Data preprossecing obj
pre = DataPE.DataPre(df)


#-----------------split & divided data------------------------

df = pre.Preprocessing()
diagnosis = df[0]
#inn = df[1] #to see the results without feature extraction
#x_train,x_test,y_train,y_test=train_test_split(inn,diagnosis, test_size=0.25, random_state=42)
independentFeatures = pre.featureExtraction()
x_train,x_test,y_train,y_test=train_test_split(independentFeatures,diagnosis,test_size=0.25,random_state=42)
y_test=y_test.astype('int64')
y_train=y_train.astype('int')


##################################################################
    
    
#---Models objects
LR = lr.LogisticRegressionModel()
SV = sv.SVMModel()
DT = dt.DecisionTree()
NB = nb.NaiveBayesModel()
#############################


#--train & accuracy
#LR.trainModel(x_train, y_train)
lrm = LR.loadSavedModel()
lrmEx = LR.loadSavedModelExt()
#LR.ModelAccuracy(y_test, lrm.predict(x_test))
LR.ModelAccuracy(y_test, lrmEx.predict(x_test))

#SV.trainModelLinear(x_train, y_train)
svmm = SV.loadSavedModel()
svmmEx = SV.loadSavedModelExt()

#----RBf
#SV.trainModelLRBF(x_train, y_train)
#RBF = SV.loadSavedRBF()
#SV.ModelAccuracyRBF(y_test, RBF.predict(x_test))

#----poly
#SV.trainModelPoly(x_train, y_train)
#poly = SV.loadPoly()
#SV.ModelAccuracyEx(y_test,poly.predict(x_test))

#----sigmoid
#SV.trainModelSigmoid(x_train, y_train)
#Sigomid = SV.loadSavedSigmoid()
#SV.ModelAccuracySigmoid(y_test, Sigomid.predict(x_test))

#----linear
#SV.ModelAccuracy(y_test,svmm.predict(x_test))
SV.ModelAccuracyEx(y_test,svmmEx.predict(x_test))



#DT.trainModel(x_train, y_train)
dtr = DT.loadSavedModel()
dtrEx = DT.loadSavedModelExt()
#DT.ModelAccuracy(y_test, dtr.predict(x_test))
DT.ModelAccuracyEx(y_test, dtrEx.predict(x_test))

#NB.trainModel(x_train, y_train)
nbm = NB.loadSavedModel()
nbmEx = NB.loadSavedModelExt()
#NB.ModelAccuracy(y_test, nbm.predict(x_test))
NB.ModelAccuracy(y_test,nbmEx.predict(x_test))


######################
#C:\Users\USER\Desktop\AI project (2)\testCSV.csv

datapath = str(input("please Enter the CSV file paht: "))
user = UserPro.user(lrmEx, svmmEx, dtrEx,nbmEx)
user.PreUserData(datapath,'E')
user.fun_of_predict()
#user = UserPro.user(lrm, svmm, dtr,nbm)
#user.PreUserData(datapath)
#user.fun_of_predict()
user.TrainVotingModule(x_train, y_train)
user.VotingModule()
