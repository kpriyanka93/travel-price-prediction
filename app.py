'''
Docstring for app
# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# BASE DIRECTORY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# LOAD ARTIFACTS

model_path = os.path.join(BASE_DIR, "models", "flight_price_main.joblib")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoders.joblib")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.joblib")

# Check files exist
for f in [model_path, encoder_path, scaler_path]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found!")

# Load artifacts
model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)
scaler = joblib.load(scaler_path)


# FEATURE COLUMNS

FEATURE_COLUMNS = ["from", "to", "flightType", "agency", "month", "year"]


# PREDICTION FUNCTION

def predict_price(input_data):
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in ["from", "to", "flightType", "agency"]:
        le = label_encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    # Convert numericals
    df[["month", "year"]] = df[["month", "year"]].astype(float)

    # Order features exactly like training
    df = df[FEATURE_COLUMNS]

    # Scale
    X_scaled = scaler.transform(df.values)

    return float(model.predict(X_scaled)[0])


# ROUTES

@app.route("/", methods=["GET"])
def home():
    return "Flight Price Prediction API is running! Use POST /predict to get predictions."

@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON input
    data = request.get_json(force=True)

    # Ensure all required keys exist
    required_keys = ["from", "to", "flightType", "agency", "month", "year"]
    if not all(k in data for k in required_keys):
        return jsonify({"error": f"Missing keys. Required: {required_keys}"}), 400

    prediction = predict_price(data)
    return jsonify({"predicted_price": round(prediction, 2)})


# RUN APP

if __name__ == "__main__":
    # Port 8000 for Docker/K8s compatibility
    app.run(host="0.0.0.0", port=8000)
'''
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

'''
model = joblib.load('C:/voyage_analytics/flight_price_app/models/flight_price_main.pkl')
label_encoders = joblib.load('C:/voyage_analytics/flight_price_app/models/label_encoders.pkl')
scaler = joblib.load('C:/voyage_analytics/flight_price_app/models/std_regressor.pkl')
'''

'''
model = joblib.load("C:/voyage_analytics/flight_price_app/models/flight_price_main.pkl")
label_encoders = joblib.load("C:/voyage_analytics/flight_price_app/models/label_encoders.joblib")
scaler = joblib.load("C:/voyage_analytics/flight_price_app/models/std_regressor.joblib")

joblib.dump(model, "C:/voyage_analytics/flight_price_app/models/flight_price_main.pkl")
joblib.dump(label_encoders, "C:/voyage_analytics/flight_price_app/models/label_encoders.joblib")
joblib.dump(scaler, "C:/voyage_analytics/flight_price_app/models/std_regressor.joblib")
'''

# some changes to address errors during building docker image
#model = joblib.load("flight_price_main.pkl")
#label_encoders = joblib.load("label_encoders.joblib")
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
'''
model_path = os.path.join(BASE_DIR, "flight_price_main.joblib")
encoder_path = os.path.join(BASE_DIR, "label_encoders.joblib")
'''


model_path = os.path.join(BASE_DIR, "models", "flight_price_main.joblib")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoders.joblib")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.joblib")
model = joblib.load(model_path)
label_encoders= joblib.load(encoder_path)
scaler = joblib.load(scaler_path)

FEATURE_COLUMNS = ["from", "to", "flightType", "agency", "month", "year"]

# PREDICTION FUNCTION

def predict_price(input_data):
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in ["from", "to", "flightType", "agency"]:
        le = label_encoders[col]
        df[col] = df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    # Convert numericals
    df[["month", "year"]] = df[["month", "year"]].astype(float)

    # Order features exactly like training
    df = df[FEATURE_COLUMNS]

    # Scale
    X_scaled = scaler.transform(df.values)

    return float(model.predict(X_scaled)[0])

# ROUTES
@app.route('/', methods=['GET', 'POST'])
def predict():
    return """

<!DOCTYPE html>
<html>

<head>
    <title>Flight Price Prediction</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        h1 {
            color: #39dde0;
            font-size: 36px;
            margin-bottom: 20px;
        }

        form {
            text-align: left;
        }

        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 15px;
            margin: 15px 0;
            border: none;
            border-bottom: 2px solid #39dde0;
            font-size: 18px;
            background-color: transparent;
            color: #39dde0;
            transition: border-bottom 0.3s ease;
        }

        input[type="text"]:focus,
        input[type="number"]:focus {
            border-bottom: 2px solid #39dde0;
            outline: none;
        }

        input[type="radio"] {
            margin-right: 10px;
        }

        input[type="submit"] {
            background-color: #39dde0;
            color: #fff;
            padding: 15px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 20px;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #39dde0;
        }

        p#prediction {
            margin-top: 20px;
            font-size: 24px;
            color: #39dde0;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Flight Price Prediction</h1>
        <hr style="border: 1px solid #39dde0; width: 80%; margin: 20px auto;">
        <form action="/predict" method="POST">

            <label>Departure City:</label><br><br>


            <input type="radio" name="From" value="Aracaju (SE)">Aracaju (SE)<br>
            <input type="radio" name="From" value="Brasilia (DF)">Brasilia (DF)<br>
            <input type="radio" name="From" value="Campo Grande (MS)">Campo Grande (MS)<br>
            <input type="radio" name="From" value="Florianopolis (SC)">Florianopolis (SC)<br>
            <input type="radio" name="From" value="Natal (RN)">Natal (RN)<br>
            <input type="radio" name="From" value="Recife (PE)">Recife (PE)<br>
            <input type="radio" name="From" value="Rio de Janeiro (RJ)">Rio de Janeiro (RJ)<br>
            <input type="radio" name="From" value="Salvador (BH)">Salvador (BH)<br>
            <input type="radio" name="From" value="Sao Paulo (SP)">Sao Paulo (SP)<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">

            <label>Destination City:</label><br><br>

            <input type="radio" name="To" value="Aracaju (SE)">Aracaju (SE)<br>
            <input type="radio" name="To" value="Brasilia (DF)">Brasilia (DF)<br>
            <input type="radio" name="To" value="Campo Grande (MS)">Campo Grande (MS)<br>
            <input type="radio" name="To" value="Florianopolis (SC)">Florianopolis (SC)<br>
            <input type="radio" name="To" value="Natal (RN)">Natal (RN)<br>
            <input type="radio" name="To" value="Recife (PE)">Recife (PE)<br>
            <input type="radio" name="To" value="Rio de Janeiro (RJ)">Rio de Janeiro (RJ)<br>
            <input type="radio" name="To" value="Salvador (BH)">Salvador (BH)<br>
            <input type="radio" name="To" value="Sao Paulo (SP)">Sao Paulo (SP)<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">

            <label>Flight Type:</label><br><br>

            <input type="radio" name="flightType" value="economic"> Economic<br>
            <input type="radio" name="flightType" value="firstClass"> FirstClass<br>
            <input type="radio" name="flightType" value="premium"> Premium<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">


            <label>Agency:</label><br><br>


            <input type="radio" name="agency" value="CloudFy"> CloudFy<br>
            <input type="radio" name="agency" value="FlyingDrops"> FlyingDrops<br>
            <input type="radio" name="agency" value="Rainbow"> Rainbow<br><br>

            <hr style="border: 1px solid #39dde0; width: 100%; margin: 20px auto;">

            <label for="weekday_num">Weekday (0=Sunday, 6=Saturday):</label>
            <input type="number" name="weekday_num" min="0" max="6" placeholder="Day of the week"><br>

            <label for="month">Month:</label>
            <input type="number" name="month" min="1" max="12" placeholder="Month"><br>

            <label for="year">Year:</label>
            <input type="number" name="year" min="2019" max="2123" placeholder="Year"><br>


            <input type="submit" value="Predict">
        </form>
        <p id="prediction"></p>
    </div>
</body>

</html>


    """

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get input data from the form
        From = request.form.get('From')
        To = request.form.get('To')
        flighttype = request.form.get('flightType')
        agency = request.form.get('agency')
        weekday_num = request.form.get('weekday_num')
        month = request.form.get('month')
        year = request.form.get('year')

        # Create a dictionary to store the input data
        input_data = {
            'from': From,
            'to': To,
            'flightType': flighttype,
            'agency': agency,
            'weekday_num': weekday_num,
            'month': month,
            'year': year
        
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(input_data)
        prediction = str(prediction)

        return jsonify({'Your Flight Price($) will be around': prediction})

# Open a tunnel on the default port 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000) 