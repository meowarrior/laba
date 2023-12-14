from fastapi import Form
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI
import numpy as np
from keras.models import load_model
import tensorflow
app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = load_model('my_model.keras')


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, sepal_length: float = Form(...), sepal_width: float = Form(...),
            petal_length: float = Form(...), petal_width: float = Form(...)):  # Добавляем параметр request
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    class_labels = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    predicted_class_label = class_labels[predicted_class]

    return templates.TemplateResponse("result.html", {"request": request, "predicted_class_label": predicted_class_label})







