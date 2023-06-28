from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form

    #City = str(data['City'])
    #type1 = str(data['type'])
    #condition = str(data['condition'])
    Area = int(data['Area'])
    floor = int(data['floor'])
    hasElevator = int(data.get('hasElevator', 0))
    hasParking = int(data.get('hasParking', 0))
    hasAirCondition = int(data.get('hasAirCondition', 0))
    hasBalcony = int(data.get('hasBalcony', 0))
    
#We couldn't deal with the categorical values in the model prediction, 
#so we removed them in order to allow the prediction to run properly.

    # categories = ["פתח תקווה","נתניה","באר שבע","הרצליה","אריאל","דימונה","רחובות","גבעת שמואל","ירושלים","שוהם","כפר סבא","רעננה","נהריה","זכרון יעקב","קרית ביאליק","חיפה","הוד השרון","תל אביב","ראשון לציון","יהוד מונוסון","נס ציונה","אילת","חולון","מודיעין מכבים רעות","צפת","בת ים","רמת גן","נוף הגליל","בית שאן"]
    # cat_type=["דירה","בית פרטי","דירת גן","דירת גג","קוטג'","דופלקס","פנטהאוז","מגרש"]
    # cat_condition=["משופץ","שמור","חדש","לא צויין","ישן"]

    # Create a feature array
    # data = {'City': [City], 'type': [type1], 'condition': [condition],'Area': [Area],'floor': [floor],
    #         'hasElevator': [hasElevator], 'hasParking': [hasParking],
    #         'hasAirCondition': [hasAirCondition],'hasBalcony': [hasBalcony]
    #         }
    data = {'Area': Area,'floor': floor, 'hasParking': hasParking,'hasElevator': hasElevator,'hasBalcony': hasBalcony,
            'hasAirCondition': hasAirCondition,'hasElevator': hasElevator
            }
    df = pd.DataFrame(data, index=[0])




    # Make a prediction
    y_pred = round(model.predict(df)[0])

    # Return the predicted price
    return render_template('index.html', price=y_pred)

if __name__ == '__main__':
    app.run()