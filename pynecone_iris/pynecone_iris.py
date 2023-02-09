"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import pynecone as pc
import pickle
import numpy as np

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"

# Loading model to compare the results
model_lr = pickle.load(open('model_lr.pkl','rb'))
x = np.array([6.2,3.4,5.4,2.3])
x = x.reshape(1,-1)
y = model_lr.predict(x)[0]
print(y)
class State(pc.State):
    """The app state."""

    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float 

    species: str = ""

    def set_sepal_length(self, sepal_length):
        self.sepal_length = float(sepal_length)
    
    def set_sepal_width(self, sepal_width):
        self.sepal_width = float(sepal_width)
    
    def set_petal_length(self, petal_length):
        self.petal_length = float(petal_length)
    
    def set_petal_width(self, petal_width):
        self.petal_width = float(petal_width)

    def get_prediction(self):
        """Get the prediction"""
        model_lr = pickle.load(open('model_lr.pkl','rb'))

        x = np.array([self.sepal_length, self.sepal_width, self.petal_length, self.petal_width])
        print(x)
        x = x.reshape(1,-1)
        self.species = model_lr.predict(x)[0]
        





# class NumberInputState(pc.State):
#     number: float

def index():
    return pc.center(
        pc.vstack(
            pc.heading("Iris Flower Classification", font_size="24px"),
            pc.form_control(
                pc.form_label("Sepal Length"),
                pc.number_input(placeholder="Enter sepal length..", on_change=State.set_sepal_length),
                pc.form_label("Sepal Width"),
                pc.number_input(placeholder="Enter sepal width..",on_change=State.set_sepal_width),
                pc.form_label("Petal Length"),
                pc.number_input(placeholder="Enter petal length..",on_change=State.set_petal_length),
                pc.form_label("Petal Width"),
                pc.number_input(placeholder="Enter petal width..",on_change=State.set_petal_width),
                is_required=True,
            ),
            pc.button(
                "Predict",
                on_click=[State.get_prediction],
                width="100%",
            ),
            pc.divider(),
            pc.heading(State.species, font_size="20px"),
        ),
        
        padding_top="10%",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
