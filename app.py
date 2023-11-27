from flask import Flask, render_template, request
import pickle

# Diccionarios de mapeo de valores numéricos a texto
airline_mapping = {
    1.000000: 'Vistara',
    0.738096: 'Air_India',
    0.046878: 'Indigo',
    0.059339: 'GO_FIRST',
    0.000000: 'AirAsia',
    0.079383: 'SpiceJet',
}

source_city_mapping = {
    0.000000: 'Delhi',
    0.831958: 'Mumbai',
    0.827241: 'Bangalore',
    0.918166: 'Kolkata',
    0.395628: 'Hyderabad',
    1.000000: 'Chennai',
}

departure_time_mapping = {
    0.896027: 'Morning',
    0.804496: 'Early Morning',
    0.867088: 'Evening',
    1.000000: 'Night',
    0.645311: 'Afternoon',
    0.000000: 'Late_Night',
}

stops_text_mapping = {
    1.000000: 'one',
    0.000000: 'zero',
    0.350277: 'two or more',
}

arrival_time_mapping = {
    0.876048: 'Night',
    1.000000: 'Evening',
    0.930839: 'Morning',
    0.613097: 'Afternoon',
    0.315340: 'Early Morning',
    0.000000: 'Late_Night',
}

destination_city_mapping = {
    0.000000: 'Delhi',
    0.833363: 'Mumbai',
    0.896218: 'Bangalore',
    1.000000: 'Kolkata',
    0.565147: 'Hyderabad',
    0.998230: 'Chennai',
}

class_mapping = {
    0.000000: 'Economy',
    1.000000: 'Business',
}

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
    # Obtener el texto asociado a los valores numéricos utilizando los diccionarios de mapeo
    airline_text = airline_mapping.get(airline, 'Valor no encontrado')
    source_city_text = source_city_mapping.get(source_city, 'Valor no encontrado')
    departure_time_text = departure_time_mapping.get(departure_time, 'Valor no encontrado')
    stops_text = stops_text_mapping.get(stops, 'Valor no encontrado')
    arrival_time_text = arrival_time_mapping.get(arrival_time, 'Valor no encontrado')
    destination_city_text = destination_city_mapping.get(destination_city, 'Valor no encontrado')
    class_text = class_mapping.get(classr, 'Valor no encontrado')
    
    # Valores originales mínimos y máximos del precio
    min_price = 1105
    max_price = 123071
    
    output = round(prediction[0], 2)
    
    new_output = output * (max_price - min_price) + min_price
    return render_template('index.html', prediction_text = f"A flight ticket going on the airline {airline_text} departing from the city {source_city_text} to go to the city of {destination_city_text} \
        on a {class_text} class flight, departing at {departure_time_text}, with {stops_text} stops and arriving {arrival_time_text}, has an estimated price of {new_output}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)