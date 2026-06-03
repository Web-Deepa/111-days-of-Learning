from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

train_mode = False 

if train_mode:
    # 1. Load data
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # 2. Define features and target 
    x = df[['Pclass', 'Sex', 'Age', 'Fare']].fillna(0) #  features
    x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
    y = df['Survived']

    # 3. Define and train pipeline 
    pipe = Pipeline([('model', RandomForestClassifier())])
    pipe.fit(x, y)
    
    # 4. Save model
    joblib.dump(pipe, "titanic_model.pkl")
    print("Model Saved")

else:
    app = Flask(__name__)
    model = joblib.load("titanic_model.pkl")
    
    @app.route('/predict_api', methods=["POST"])
    def predict():
        data = request.get_json()
        passenger = pd.DataFrame([data])
        
        # 5.Make predictions
        pred = model.predict(passenger)[0]
        prob = model.predict_proba(passenger)[0][1] 
        
        return jsonify({
            'survived': int(pred),
            'probability': round(float(prob), 3), 
            'result': 'Survived' if pred == 1 else 'Not Survived'
        })

    if __name__ == '__main__':
        app.run(debug=True)
