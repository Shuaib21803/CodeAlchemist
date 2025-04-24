# CodeAlchemist: Sklearn Model Deployment and Sharing Platform

CodeAlchemist is a platform that allows users to deploy and share sklearn models easily. It provides a simple way to serve machine learning models and generate shareable links for testing.

## Features

- Deploy sklearn models with ease
- Generate shareable links for deployed models
- Allow friends to test models through a user-friendly interface
- Support for various sklearn model types
- Example implementation using a Random Forest Regressor for advertising sales prediction

## Project Structure

- `01_Model_Preparation/`: Contains example notebook for model preparation
- `02_Prediction_API/`: Houses the Flask-based API for serving predictions
- `code-alchemist/`: Frontend code for the web interface
- `requirements.txt`: List of Python dependencies
- `streamlit_local.py`: Local Streamlit interface for quick testing

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/CodeAlchemist.git
   cd CodeAlchemist
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv Scripts activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Deploying a Model

1. Prepare your sklearn model and save it using joblib or pickle.
2. Place your model file in the `02_Prediction_API/models/` directory.
3. Update the `02_Prediction_API/Prediction_API.py` file to load and use your model.
4. Start the Flask server:
   ```
   python -m flask --app 02_Prediction_API/Prediction_API run
   ```

### Sharing Your Model

Once your model is deployed, you can share the API endpoint with others. They can use this endpoint to make predictions using your model.

### Making Predictions

To make predictions using the API:

1. Send a POST request to `http://your-server-address/api` with JSON data in the format required by your model.
2. The API will return the prediction results.

Example using curl:
```
curl -X POST http://localhost:5000/api \
     -H "Content-Type: application/json" \
     -d '[{"feature1": value1, "feature2": value2, ...}]'
```

### Local Testing with Streamlit

For quick local testing, you can use the Streamlit interface:

1. Run the Streamlit app:
   ```
   streamlit run streamlit_local.py
   ```
2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).
3. Input the required features and click "Predict" to see the model's output.

## Customization

To use CodeAlchemist with your own sklearn model:

1. Replace the example model in `01_Model_Preparation/` with your own model preparation notebook.
2. Update the `02_Prediction_API/Prediction_API.py` to load and use your specific model.
3. Modify the Streamlit interface in `streamlit_local.py` to match your model's input requirements.

