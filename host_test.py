from flask import Flask,request
from tensorflow.keras.models import load_model
import pandas as pd
model=load_model(r"F:\Generative AI\RNN\bbc_model.h5")
import joblib
encoder=joblib.load(r"F:\Generative AI\RNN\label_encoder.pkl")
vec=joblib.load(r"F:\Generative AI\RNN\feature_vec.pkl")
app=Flask(__name__)
@app.route('/',methods=['POST'])
def prediction():
    data=request.get_json(force=True)
    data=data['text']
    print(data)
    encoded_data=vec.transform(data)
    output=model.predict(encoded_data.toarray())
    import numpy as np
    out_max=np.zeros_like(output)
    for i in range(output.shape[0]):
        out=np.argmax(output[i])
        out_max[i][out]=1
    y_pred=encoder.inverse_transform(out_max)
    print(y_pred)
    return str(y_pred)
app.run(host='0.0.0.0')