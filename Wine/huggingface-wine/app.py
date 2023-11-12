import gradio as gr
#from gradio.components import inputs
from PIL import Image
import requests
import hopsworks
import io
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.json")
print("Model downloaded")

def wine(volatile_acidity,
           residual_sugar,
           chlorides,
           free_sulfur_dioxide,
           alcohol):
    print("Calling function")
    df = pd.DataFrame([[volatile_acidity, residual_sugar, chlorides, free_sulfur_dioxide, alcohol]], 
                      columns=["volatile_acidity", "residual_sugar", "chlorides", "free_sulfur_dioxide", "alcohol"])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res[0])
    wine_url = "https://github.com/jordicotxet/id2223/blob/63fe7d525afa1cfb626c9fa7513e2cc886e22d41/Wine/wine_dataset/" + str(res[0]) + ".jpg?raw=true"
    print(wine_url)
    img = Image.open(requests.get(wine_url, stream=True).raw)  
    return img
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with few main wine characteristics to predict which quality it is.",
    allow_flagging="never",
    inputs=[
        gr.Slider(minimum=0, maximum=1.5, step=0.01, value=0.2, label="volatile acidity"),
        gr.Slider(minimum=0, maximum=100, step=0.1, value=5.9, label="residual sugar"),
        gr.Slider(minimum=0, maximum=0.5, step=0.001, value=0.046, label="chlorides"),
        gr.Slider(minimum=0, maximum=400, step=1, value=35, label="free_sulfur_dioxide"),
        gr.Slider(minimum=2, maximum=15, step=0.1, value=10.6, label="alcohol (in %)"),
        ],
    examples=[[0.5, 0.8, 0.034, 46, 9.2],[0.42, 4.1, 0.03, 31, 12.8], [0.7, 67.1, 0.219, 275, 10.7]],
    outputs=gr.Image(type="pil", height=400, width=400))

demo.launch(debug=True)
 
