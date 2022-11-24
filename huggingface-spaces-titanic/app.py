import gradio as gr
import numpy as np
from PIL import Image
import requests
import random

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=22)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

def titanic(Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked):

    if Pclass == 0:
        Pclass_1 = 1
        Pclass_2 = 0
        Pclass_3 = 0
    elif Pclass == 1:
        Pclass_1 = 0
        Pclass_2 = 1
        Pclass_3 = 0
    elif Pclass == 2:
        Pclass_1 = 0
        Pclass_2 = 0
        Pclass_3 = 1
    Pclass_1 = np.float64(Pclass_1)
    Pclass_2 = np.float64(Pclass_2)
    Pclass_3 = np.float64(Pclass_3)
    Pclass_1 = Pclass_1.astype(np.uint8)
    Pclass_2 = Pclass_2.astype(np.uint8)
    Pclass_3 = Pclass_3.astype(np.uint8)
        
        
    Familysize = SibSp+Parch+1
    if Familysize == 1:
        Family_Single = 1
        Family_Small = 0
        Family_Large = 0
    elif 2 <= Familysize <= 4:
        Family_Single = 0
        Family_Small = 1
        Family_Large = 0
    elif 5 <= Familysize:
        Family_Single = 0
        Family_Small = 0
        Family_Large = 1
        
    if Sex == 'female':
        Sex = 0
    elif Sex == 'male':
        Sex = 1
    Sex = np.float64(Sex)
    Sex = Sex.astype(np.uint8)

    if Cabin[0] == 'B':
        Cabin_B = 1
        Cabin_U = 0
    elif Cabin[0] == 'U':
        Cabin_B = 0
        Cabin_U = 1
    else:
        Cabin_B = 0
        Cabin_U = 0
    Cabin_U = np.float64(Cabin_U)
    Cabin_B = np.float64(Cabin_B)
    Cabin_U = Cabin_U.astype(np.uint8)
    Cabin_B = Cabin_B.astype(np.uint8)


    if Embarked == 'C':
        Embarked_C = 1
        Embarked_Q = 0
        Embarked_S = 0
    elif Embarked == 'Q':
        Embarked_C = 0
        Embarked_Q = 1
        Embarked_S = 0
    elif Embarked == 'S':
        Embarked_C = 0
        Embarked_Q = 0
        Embarked_S = 1
    Embarked_C = np.float64(Embarked_C)
    Embarked_Q = np.float64(Embarked_Q)
    Embarked_S = np.float64(Embarked_S)
    Embarked_C = Embarked_C.astype(np.uint8)
    Embarked_Q = Embarked_Q.astype(np.uint8)
    Embarked_S = Embarked_S.astype(np.uint8)
    
    input_list = []
    input_list.append(Pclass_1)
    input_list.append(Pclass_2)
    input_list.append(Pclass_3)
    input_list.append(Familysize)
    input_list.append(Family_Single)
    input_list.append(Family_Small)
    input_list.append(Family_Large)
    input_list.append(Fare)
    input_list.append(Sex)
    input_list.append(Cabin_B)
    input_list.append(Cabin_U)
    input_list.append(Embarked_C)
    input_list.append(Embarked_Q)
    input_list.append(Embarked_S)


    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    print(res[0])
    return res[0]

        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with Pclass(Ticket class), Sex, Age, SibSp(Number of siblings), Parch(Number of parents/children),Ticket, Fare(Ticket price), Cabin(cabin number),embarked(port of embarkation).",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1, label="Pclass(Ticket class 1st- 1, 2nd- 2, 3rd- 3)"),
        gr.inputs.Textbox(default='Kenneth', label="Name, e.g. Kenneth"),
        gr.inputs.Textbox(default='male', label="Sex(female or male)"),
        gr.inputs.Number(default=10, label="Age(in years)"),
        gr.inputs.Number(default=1, label="SibSp(Number of siblings)"),
        gr.inputs.Number(default=1, label="Parch(Number of parents/children)"),
        gr.inputs.Textbox(default='ID2223', label="Ticket"),
        gr.inputs.Number(default=30, label="Fare(Ticket price)"),
        gr.inputs.Textbox(default='A250', label="Cabin number"),
        gr.inputs.Textbox(default='S', label="port of embarkation"),
        ],

    outputs=gr.Number(label="Survived- 1, Died- 0")
    )
demo.launch(share=True)
