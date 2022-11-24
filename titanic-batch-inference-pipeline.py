
import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()
###############################################################################
def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=22)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    offset = 1
    survival = y_pred[y_pred.size-offset]
    
    if survival == 1:
        survival = "Survived"
        print("survival predicted: Survived.")
    else:
        survival = "Died"
        print("survival predicted: Died.")
    
    titanic_url = "https://github.com/YuhangDeng123/ID2223Lab1_TitanicSurvival/blob/main/graph/" + survival + ".jpg?raw=true"
    img = Image.open(requests.get(titanic_url, stream=True).raw)            
    img.save("./latest_titanic.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_titanic.png", "Resources/images", overwrite=True)
   
    titanic_fg = fs.get_feature_group(name="titanic_modal", version=1)
    df = titanic_fg.read() 
    label = df.iloc[-offset]["survived"]
    if label == 1:
        label = 'Survived'
        print("survival actual: Survived.")
    else:
        label = 'Died'
        print("survival actual: Died.")
    
    label_url= "https://github.com/YuhangDeng123/ID2223Lab1_TitanicSurvival/blob/main/graph/" + label + ".jpg?raw=true"
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_titanic.png")
    dataset_api.upload("./actual_titanic.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="titanic survival Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survival],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df])
    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]


    print("Number of different survival situation predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True died', 'True survived'],
                             ['Pred died', 'Pred survived'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("Something errors.")


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

