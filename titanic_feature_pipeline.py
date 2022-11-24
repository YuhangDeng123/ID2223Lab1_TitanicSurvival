
import os
import modal
import hopsworks
import pandas as pd
import numpy as np
from random import choice

project = hopsworks.login()
fs = project.get_feature_store()

original_data= pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
################################# data processing #############################

original_data['Age'] = original_data['Age'].fillna(original_data['Age'].mean())
original_data['Fare'] = original_data['Fare'].fillna(original_data['Fare'].mean())

################################ data cleaning ################################
# The Cabin has a high missing rate and is useless. Fill 'U' represents 'unknown'
original_data['Cabin'] = original_data['Cabin'].fillna('U')
# 'Embarked' has only two missing, most of which are S. Fill 'S'
original_data['Embarked'] = original_data['Embarked'].fillna('S')
# 'Name' and 'Ticket' seem useless, drop them.
list_drop = ['Name', 'Ticket']
original_data.drop(list_drop, axis=1, inplace=True)


########################## Standardization of features ########################
# One-host method, normalized to 0,1
original_data['Sex']= original_data['Sex'].map({'female':0,'male':1})


EmbarkedCODED = pd.DataFrame() 
EmbarkedCODED = pd.get_dummies(original_data['Embarked'],prefix='Embarked')
original_data = pd.concat([original_data,EmbarkedCODED],axis=1)
original_data.drop('Embarked',axis=1,inplace = True)


PclassCODED = pd.DataFrame() 
PclassCODED = pd.get_dummies(original_data['Pclass'],prefix='Pclass')
original_data = pd.concat([original_data,PclassCODED],axis=1)
original_data.drop('Pclass',axis=1,inplace = True)


CabinCODED = pd.DataFrame() 
original_data['Cabin'] = original_data['Cabin'].map(lambda c:c[0]) 
CabinCODED = pd.get_dummies(original_data['Cabin'],prefix='Cabin')
original_data = pd.concat([original_data,CabinCODED],axis=1)
original_data.drop('Cabin',axis=1,inplace = True)


familyCODED = pd.DataFrame() 
familyCODED['Familysize'] = original_data['SibSp'] + original_data['Parch'] + 1
familyCODED['Family_Single'] = familyCODED['Familysize'].map(lambda s: 1 if s==1 else 0)
familyCODED['Family_Small'] = familyCODED['Familysize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
familyCODED['Family_Large'] = familyCODED['Familysize'].map(lambda s: 1 if 5 <= s else 0)
original_data = pd.concat([original_data,familyCODED],axis=1)
list_drop = ['SibSp', 'Parch']
original_data.drop(list_drop,axis=1,inplace = True)

AgeCODED = pd.DataFrame() 
AgeCODED['Kid'] = original_data['Age'].map(lambda a: 1 if 0 < a <=6 else 0)
AgeCODED['Teens'] = original_data['Age'].map(lambda a: 1 if 6 < a <=18 else 0)
AgeCODED['Youth'] = original_data['Age'].map(lambda a: 1 if 18 < a <=40 else 0)
AgeCODED['Midaged'] = original_data['Age'].map(lambda a: 1 if 40 < a <=60 else 0)
AgeCODED['Olds'] = original_data['Age'].map(lambda a: 1 if 60 < a  else 0)
original_data = pd.concat([original_data,AgeCODED],axis=1)
original_data.drop('Age',axis=1,inplace = True)

########### select training set features and training set labels ##############
original_data_X = pd.concat([original_data['Survived'],PclassCODED,familyCODED,original_data['Fare'],original_data['Sex'],original_data['Cabin_B'],original_data['Cabin_U'],EmbarkedCODED],axis=1)

###################### upload training set to Hopsworks #######################

titanic_fg = fs.get_or_create_feature_group(
    name="titanic_modal",
    version=1,
    primary_key=['Survived',"Pclass_1","Pclass_2","Pclass_3","Familysize","Family_Single","Family_Small","Family_Large","Fare","Sex","Cabin_B","Cabin_U","Embarked_C","Embarked_Q","Embarked_S"], 
    description="Titanic survival dataset")
titanic_fg.insert(original_data_X, write_options={"wait_for_job" : False})


