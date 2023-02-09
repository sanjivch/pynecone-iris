"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import pynecone as pc
import pickle
import numpy as np

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"

# Loading model to compare the results

class State(pc.State):
    """The app state."""

    sepal_length: float = 0.0
    sepal_width: float = 0.0
    petal_length: float = 0.0
    petal_width: float = 0.0

    species: str = ""
    def get_prediction(self):
        """Get the prediction"""
        model_lr = pickle.load(open('model_lr.pkl','rb'))

        x = np.array([self.sepal_length, self.sepal_width, self.petal_length, self.petal_width])
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
                pc.number_input(placeholder="Enter sepal length..",on_blur=State.set_sepal_length),
                pc.form_label("Sepal Width"),
                pc.number_input(placeholder="Enter sepal width..",on_blur=State.set_sepal_width),
                pc.form_label("Petal Length"),
                pc.number_input(placeholder="Enter petal length..",on_blur=State.set_petal_length),
                pc.form_label("Petal Width"),
                pc.number_input(placeholder="Enter petal width..",on_blur=State.set_petal_width),
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
