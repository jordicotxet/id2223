import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine", version=1)
    query = wine_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="wine",
                                    version=1,
                                    description="Read from wine dataset",
                                    labels=["quality"],
                                    query=query)

    #feature_view = fs.get_or_create_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version = 1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    #get features added the last day
    
    y_pred = model.predict(batch_data)
    #print(y_pred)
    offset = 1
    wine_quality = int(y_pred[y_pred.size-offset])
    wine_url = "https://github.com/jordicotxet/id2223/blob/63fe7d525afa1cfb626c9fa7513e2cc886e22d41/Wine/wine_dataset/" + str(wine_quality) + ".jpg?raw=true"
    print("Quality predicted: ", wine_quality)
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)
   
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    #print(df)
    label = int(df.iloc[-offset]["quality"])
    label_url = "https://github.com/jordicotxet/id2223/blob/63fe7d525afa1cfb626c9fa7513e2cc886e22d41/Wine/wine_dataset/" + str(label) + ".jpg?raw=true"
    print("Wine actual: ",label)
    img = Image.open(requests.get(label_url, stream=True).raw)            
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="White Wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine_quality],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_wine_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_wine_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 3:
        results = confusion_matrix(labels, predictions)

        df_cm = pd.DataFrame(results, 
                                ['0', '1', '2'],
                                ['0', '1', '2'])

        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./wine_confusion_matrix.png")
        dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 3 different wine predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different wine quality predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

