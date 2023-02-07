"""Welcome to Pynecone! This file outlines the steps to create a basic app."""
from pcconfig import config

import pynecone as pc
import pickle
import numpy as np

docs_url = "https://pynecone.io/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"

# Loading model to compare the results
model_lr = pickle.load(open('model_lr.pkl','rb'))

x = np.array([4.9, 3.0, 1.4, 0.2])
x=x.reshape(1,-1)
y_pred = model_lr.predict(x)
print(y_pred)

class State(pc.State):
    """The app state."""

    pass
import random
class VarNumberState(pc.State):
    number: int

    def update(self):
        self.number = random.randint(0, 100)


class NumberInputState(pc.State):
    number: float

def index():
    return pc.center(
        pc.vstack(
            pc.heading("Iris Flower Classification", font_size="24px"),
            pc.form_control(
                pc.form_label("Sepal Length"),
                pc.number_input(on_change=NumberInputState.set_number,),
                pc.form_label("Sepal Length"),
                pc.number_input(on_change=NumberInputState.set_number,),
                pc.form_label("Sepal Length"),
                pc.number_input(on_change=NumberInputState.set_number,),
                pc.form_label("Sepal Length"),
                pc.number_input(on_change=NumberInputState.set_number,),
                is_required=True,
            ),
            pc.heading(y_pred[0], font_size="20px"),
        ),
        
        padding_top="10%",
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index)
app.compile()
