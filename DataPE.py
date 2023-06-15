from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np



class DataPre:
    
 def __init__(self, df):
     self.df = df
#---------------------Data Preprocessing--------------------\
 def Preprocessing(self):  
  self.df = self.df.drop(['Index'], axis =1)   
  self.df = self.df.dropna()
        
# print(df.duplicated()) =false ->there is no duplicated rows
# drop duplicate code -> Keep the last occurance of the duplicated row and remove others in the set
  self.df = self.df.drop_duplicates(keep="last")


#drow outlierss
  #sns.boxplot(data=self.df)
  
 

# B-> negative      M->positive
# replace diagnosis column with 0->B  1->M
  self.df.loc[self.df["diagnosis"] == "B", "diagnosis"] = 0
  self.df.loc[self.df["diagnosis"] == "M", "diagnosis"] = 1
  
  #-----data Scaling
  impData = self.df.drop(['diagnosis'], axis = 1)
  scaler = StandardScaler()
  scaler.fit(impData)
  self.scaled_data = scaler.transform(impData)
  
 
  return self.df["diagnosis"],self.scaled_data

#----------------------Feature Extraction

 def featureExtraction(self):
      

         #-----Feature Extraction
   pca = PCA(n_components=3)
   pca.fit(self.scaled_data)
   pca_data = pca.transform(self.scaled_data)
   return pca_data