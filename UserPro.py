from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import DataPE



class user:
    def __init__(self, LR,SV,DT,NB):
        self.LR = LR
        self.SV = SV
        self.DT = DT
        self.NB = NB

 
#------------------predect-----------------        
    def PreUserData(self, path,typ='B'):
        
        self.InputData= pd.read_csv(path)
        pre = DataPE.DataPre(self.InputData)
        self.InputData= pre.Preprocessing()
        self.y_userData = self.InputData[0]
        if(typ == 'E'):
            self.InputData= pre.featureExtraction()
            
            
        else:
            self.InputData = self.InputData[1]
            
            
    def fun_of_predict(self):
        self.prediction_of_LR  = self.LR.predict(self.InputData)
        self.prediction_of_SVM = self.SV.predict(self.InputData)
        self.prediction_of_DT  = self.DT.predict(self.InputData)
        self.prediction_of_NB = self.NB.predict(self.InputData)
        print("logisticModel:")
        print(self.prediction_of_LR)
        print("SVMmodel:")
        print(self.prediction_of_SVM)
        print("DecisionTreeModel:")
        print(self.prediction_of_DT)
        print("NaiveBayesModel:")
        print(self.prediction_of_NB)
        
        
    #----------------------------Voting Modula------------------------------------ 
    def TrainVotingModule(self,x_train, y_train):
        self.voting_cls_hard = VotingClassifier(estimators= [('logistic',self.LR),
                                                    ('DecisionTree',self.DT), 
                                                    ('SVM', self.SV),
                                                    ('Naive', self.NB)], voting="hard")


        self.voting_cls_hard.fit(x_train,y_train)
       
            

    def VotingModule(self):
        
        predect = self.voting_cls_hard.predict(self.InputData)

        lisst = []
        for x in predect:
            if(x == 0):
                lisst.append("B")
            else :
                lisst.append("M")
          
        print("")      
        print("Data-Final Predection:")
        print(lisst) 

        print("predection accuracy: ")   
        score=accuracy_score(list(self.y_userData),predect)*100
        print(score)


    def voting_builtin(self):
        listOfPred = []
        listt =[]
        for (x,y,z,b) in zip(self.prediction_of_LR,  self.prediction_of_SVM, self.prediction_of_DT,self.prediction_of_NB):
            B = 0
            M = 0
            if x == 0:
                B+=1
            else:
                M+=1
            if y == 0:
                B+=1
            else:
                M+=1
            if z == 0:
                B+=1
            else:
                M+=1
            if b == 0:
                B+=1
            else:
                M+=1    
                
            if B > M :
                listOfPred.append(0)
            else:
                listOfPred.append(1)
                
        score=accuracy_score(list(self.y_userData),listOfPred)*100
        for x in listOfPred:
               if(x == 0):
                   listt.append("B")
               else :
                  listt.append("M")
        print("")        
        print(listt)
        print(score)         
   