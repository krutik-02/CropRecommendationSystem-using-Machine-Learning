from flask import Flask, request, render_template
import numpy as np
import pickle
import requests

# Importing the model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standardscaler.pkl', 'rb'))

API_KEY = '5a4aa785bacc5c7a8a750e2e41988507'

# Creating the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/manual')
def index():
    return render_template('manual.html')

@app.route('/manual', methods=['POST'])
def predict():
    # Retrieve form data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']
    
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    final_features = sc.transform(single_pred)
    prediction = model.predict(final_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 
        10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    
    # Mapping of crop names to image filenames
    crop_images = {
        "Rice": "rice.jpg",
        "Maize": "maize.jpg",
        "Jute": "jute.jpg",
        "Cotton": "cotton.jpg",
        "Coconut": "coconut.jpg",
        "Papaya": "papaya.jpg",
        "Orange": "orange.jpg",
        "Apple": "apple.jpg",
        "Muskmelon": "muskmelon.jpg",
        "Watermelon": "watermelon.jpg",
        "Grapes": "grapes.jpg",
        "Mango": "mango.jpg",
        "Banana": "banana.jpg",
        "Pomegranate": "pomegranate.jpg",
        "Lentil": "lentil.jpg",
        "Blackgram": "blackgram.jpg",
        "Mungbean": "mungbean.jpg",
        "Mothbeans": "mothbeans.jpg",
        "Pigeonpeas": "pigeonpeas.jpg",
        "Kidneybeans": "kidneybeans.jpg",
        "Chickpea": "chickpea.jpg",
        "Coffee": "coffee.jpg",
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there.".format(crop)
        crop_image = crop_images.get(crop)  # Get the corresponding image filename
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        crop_image = None  # No image to display

    return render_template('manual.html', result=result, crop_image=crop_image)

# To take input based on the region
@app.route('/region_select')
def region():
    return render_template('region.html')

@app.route('/region_select', methods=['POST'])
def predict_region():
    # Get user input from the form
    region = request.form['Region']
    nitrogen = request.form['Nitrogen']
    phosphorus = request.form['Phosphorus']
    potassium = request.form['Potassium']
    ph = request.form['Ph']

    # Fetch weather data for the region from OpenWeatherMap API (Current weather)
    weather_params = {'q': region, 'appid': API_KEY, 'units': 'metric'}
    weather_data = requests.get("http://api.openweathermap.org/data/2.5/weather", params=weather_params).json()

    if weather_data.get('cod') == 200:  # Check if the request was successful
        # Extract weather dat++a
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        rainfall = weather_data.get('rain', {}).get('1h', 0)  # Get rainfall data if available

        # Crop Prediction Logic (using the model)
        feature_list = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        final_features = sc.transform(single_pred)
        prediction = model.predict(final_features)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        # Mapping of crop names to image filenames
        crop_images = {
            "Rice": "rice.jpg",
            "Maize": "maize.jpg",
            "Jute": "jute.jpg",
            "Cotton": "cotton.jpg",
            "Coconut": "coconut.jpg",
            "Papaya": "papaya.jpg",
            "Orange": "orange.jpg",
            "Apple": "apple.jpg",
            "Muskmelon": "muskmelon.jpg",
            "Watermelon": "watermelon.jpg",
            "Grapes": "grapes.jpg",
            "Mango": "mango.jpg",
            "Banana": "banana.jpg",
            "Pomegranate": "pomegranate.jpg",
            "Lentil": "lentil.jpg",
            "Blackgram": "blackgram.jpg",
            "Mungbean": "mungbean.jpg",
            "Mothbeans": "mothbeans.jpg",
            "Pigeonpeas": "pigeonpeas.jpg",
            "Kidneybeans": "kidneybeans.jpg",
            "Chickpea": "chickpea.jpg",
            "Coffee": "coffee.jpg",
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated in {}.".format(crop, region)
            crop_image = crop_images.get(crop)  # Get the corresponding image filename
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            crop_image = None  # No image to display

        # Pass weather data to the template
        weather_info = {
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall
        }

        return render_template('region.html', result=result, crop_image=crop_image, weather=weather_info)
    
    else:
        result = "Sorry, we couldn't fetch the weather data for the provided region."
        return render_template('region.html', result=result)


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
