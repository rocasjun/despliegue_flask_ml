from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model_lr = pickle.load(open('linear_regression_model.pkl', 'rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    airline = float (request.form['airline'])
    source_city = float (request.form['source_city'])
    departure_time = float (request.form['departure_time'])
    stops = float (request.form['stops'])
    arrival_time = float (request.form['arrival_time'])
    destination_city = float (request.form['destination_city'])
    classr = float (request.form['classr'])
    duration = float (request.form['duration'])
    days_left = float (request.form['days_left'])
    
    prediction = model_lr.predict([[
    airline, source_city, departure_time, stops, arrival_time, destination_city,
    classr, duration, days_left
]])
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text = f"Un ticket de vuelo que va desde {source_city}, que sale a las {departure_time} y llega a las {arrival_time} \
        con {stops} paradas, tiene un precio estimado de {output}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)