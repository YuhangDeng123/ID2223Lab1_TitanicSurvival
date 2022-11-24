
import os
import modal
import numpy as np

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("jim-hopsworks-ai"))
   def f():
       g()
###############################################################################

def generate_passenger(Survived, 
                       P_Pclass_1, P_Pclass_2, P_Pclass_3, 
                       P_Family_1, P_Family_2, P_Family_3, 
                       P_Cabin_B, P_Cabin_U,
                       P_Embarked_C, P_Embarked_Q, P_Embarked_S, 
                       fare_len_max, fare_len_min):

    import pandas as pd
    import random
    Pclass = random.uniform(0, 1)
    if Pclass >= P_Pclass_1:
        Pclass_1 = 1
        Pclass_2 = 0
        Pclass_3 = 0
    elif Pclass >= P_Pclass_2:
        Pclass_1 = 0
        Pclass_2 = 1
        Pclass_3 = 0
    elif Pclass >= P_Pclass_3:
        Pclass_1 = 0
        Pclass_2 = 0
        Pclass_3 = 1

    Pclass_1 = np.float64(Pclass_1)
    Pclass_2 = np.float64(Pclass_2)
    Pclass_3 = np.float64(Pclass_3)
    Pclass_1 = Pclass_1.astype(np.uint8)
    Pclass_2 = Pclass_2.astype(np.uint8)
    Pclass_3 = Pclass_3.astype(np.uint8)
        
        
    Family = random.uniform(0, 1)
    if Family >= P_Family_1:
        Familysize = 1
        Family_Single = 1
        Family_Small = 0
        Family_Large = 0
    elif Family >= P_Family_2:
        Familysize = random.choice([2,3,4])
        Family_Single = 0
        Family_Small = 1
        Family_Large = 0
    elif Family >= P_Family_3:
        Familysize = random.choice([5,6,7,8])
        Family_Single = 0
        Family_Small = 0
        Family_Large = 1
        
    Cabin = random.uniform(0, 1)
    if Cabin >= P_Cabin_B:
        Cabin_B = 1
        Cabin_U = 0
    elif Cabin >= P_Cabin_U:
        Cabin_U = 1
        Cabin_B = 0
        
    Cabin_U = np.float64(Cabin_U)
    Cabin_B = np.float64(Cabin_B)
    Cabin_U = Cabin_U.astype(np.uint8)
    Cabin_B = Cabin_B.astype(np.uint8)

    Embarked = random.uniform(0, 1)
    if Embarked >= P_Embarked_C:
        Embarked_C = 1
        Embarked_Q = 0
        Embarked_S = 0
    elif Embarked >= P_Embarked_Q:
        Embarked_C = 0
        Embarked_Q = 1
        Embarked_S = 0
    elif Embarked >= P_Embarked_S:
        Embarked_C = 0
        Embarked_Q = 0
        Embarked_S = 1

    Embarked_C = np.float64(Embarked_C)
    Embarked_Q = np.float64(Embarked_Q)
    Embarked_S = np.float64(Embarked_S)
    Embarked_C = Embarked_C.astype(np.uint8)
    Embarked_Q = Embarked_Q.astype(np.uint8)
    Embarked_S = Embarked_S.astype(np.uint8)
        
    df = pd.DataFrame({ "pclass_1": [Pclass_1],
                       "pclass_2": [Pclass_2],
                       "pclass_3": [Pclass_3],
                       "familysize": [Familysize],
                       "family_Single": [Family_Single],
                       "family_Small": [Family_Small],
                       "family_Large": [Family_Large],
                       "fare": [random.uniform(fare_len_max, fare_len_min)],
                       "sex": [random.choice([0,1])],
                       "cabin_B": [Cabin_B],
                       "cabin_U": [Cabin_U],
                       "embarked_C": [Embarked_C],
                       "embarked_Q": [Embarked_Q],
                       "embarked_S": [Embarked_S],
                      })
    df['Survived'] = Survived
    return df


def get_random_titanic_passenger():

    import pandas as pd
    import random

    survived_df = generate_passenger(1, 0.3, 0.1, 0, 0.9, 0.2, 0, 0.3, 0, 0.3, 0.2, 0, 200, 100)
    dead_df = generate_passenger(0, 0.9, 0.7, 0, 0.4, 0.35, 0, 0.9, 0, 0.7, 0.6, 0, 100, 5)

    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = survived_df
        print("survived_passenger added")
    else:
        passenger_df = dead_df
        print("dead_passenger added")

    return passenger_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    passenger_df = get_random_titanic_passenger()
    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    print("inserting")
    print(passenger_df)
    titanic_fg.insert(passenger_df, write_options={"wait_for_job" : False})
    print("inserted")

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
