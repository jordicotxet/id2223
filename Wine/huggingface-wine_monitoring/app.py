import gradio as gr
import hopsworks
from PIL import Image

project = hopsworks.login()

class ImageLoad:
    def __init__(self, path, project) -> None:
        self.path = path
        self.image_name = path[path.rfind('/') + 1:]
        self.project = project

    def __call__(self):
        dataset_api = self.project.get_dataset_api()
        dataset_api.download(self.path, overwrite=True)
        return Image.open(self.image_name)



with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Image")
          input_img = gr.Image(value=ImageLoad("Resources/images/latest_wine.png", project), elem_id="predicted-img")
      with gr.Column():          
          gr.Label("Today's Actual Image")
          input_img = gr.Image(value=ImageLoad("Resources/images/actual_wine.png", project), elem_id="actual-img")        
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image(value=ImageLoad("Resources/images/df_wine_recent.png", project), elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")  
          input_img = gr.Image(value=ImageLoad("Resources/images/wine_confusion_matrix.png", project), elem_id="recent-predictions")

demo.launch(share = True)
