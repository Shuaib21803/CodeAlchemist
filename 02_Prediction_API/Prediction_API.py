from flask import Flask, request, jsonify
import joblib
import pandas as pd

model = joblib.load('/home/tanmay/Desktop/CodeAlchemist/02_Prediction_API/models/updated_production_model.pkl')
col_names = joblib.load('/home/tanmay/Desktop/CodeAlchemist/02_Prediction_API/models/updated_production_model_column_names.pkl')

app = Flask(__name__)

# wait for json post request and return prediction
@app.route('/api', methods=['POST'])
def predict():
    req_data = request.json
    df = pd.DataFrame(req_data).reindex(columns=col_names)

    prediction = list(model.predict(df))

    return jsonify({'prediction': str(prediction)})

# load the model and start web api
if __name__ == 'main':
    app.run()
