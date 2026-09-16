"""
CSE274: Applied Machine Learning
Student Performance Risk Prediction — Flask API
"""
from flask import Flask, request, jsonify, send_from_directory
import joblib, numpy as np, os
import pandas as pd

app = Flask(__name__, static_folder='static')

pipeline     = joblib.load('model.pkl')
le_dict      = joblib.load('label_encoders.pkl')
feature_cols = joblib.load('feature_cols.pkl')
label_maps   = joblib.load('label_maps.pkl')

NUMERICAL   = ['Hours_Studied','Attendance','Sleep_Hours','Previous_Scores',
               'Tutoring_Sessions','Physical_Activity']
CATEGORICAL = ['Parental_Involvement','Access_to_Resources','Extracurricular_Activities',
               'Motivation_Level','Family_Income','Teacher_Quality',
               'Parental_Education_Level','Distance_to_School','Gender']

def recommend_hours(data, risk_prob):
    current = float(data.get('Hours_Studied', 10))
    attend  = float(data.get('Attendance', 75))
    prev    = float(data.get('Previous_Scores', 65))
    if   risk_prob >= 0.7: rec = max(current * 1.5, 25)
    elif risk_prob >= 0.4: rec = max(current * 1.2, 18)
    else:                  rec = max(current, 15)
    if attend < 70: rec += 3
    if prev   < 60: rec += 4
    return int(min(round(rec), 50))

def get_tips(risk_prob, data):
    tips = []
    if float(data.get('Hours_Studied', 20)) < 15:
        tips.append("📚 Study at least 3 hours/day — consistency beats cramming.")
    if float(data.get('Attendance', 80)) < 75:
        tips.append("🏫 Improve attendance — every missed class creates a knowledge gap.")
    if float(data.get('Sleep_Hours', 7)) < 6:
        tips.append("😴 Target 7–8 hours of sleep for better memory retention.")
    if float(data.get('Tutoring_Sessions', 0)) == 0 and risk_prob > 0.4:
        tips.append("👨‍🏫 Enroll in tutoring sessions to strengthen weak subjects.")
    if float(data.get('Physical_Activity', 2)) < 2:
        tips.append("🏃 Add 30 min of daily exercise — boosts focus and mood.")
    if data.get('Motivation_Level') == 'Low':
        tips.append("🎯 Set small daily goals to rebuild study motivation.")
    if not tips:
        tips.append("✅ Great profile! Keep up consistent effort and regular revision.")
    return tips

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('plots', filename)

@app.route('/api/options')
def options():
    return jsonify(label_maps)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = []
        for col in feature_cols:
            if col in NUMERICAL:
                features.append(float(data.get(col, 0)))
            else:
                le  = le_dict[col]
                val = data.get(col, le.classes_[0])
                features.append(int(le.transform([val])[0]) if val in le.classes_ else 0)

        X = pd.DataFrame([features], columns=feature_cols)
        risk_prob = float(pipeline.predict_proba(X)[0][1])
        pred      = int(pipeline.predict(X)[0])

        if   risk_prob >= 0.7: risk_level, color = "High Risk",     "red"
        elif risk_prob >= 0.4: risk_level, color = "Moderate Risk", "orange"
        else:                  risk_level, color = "Low Risk",      "green"

        return jsonify({
            'prediction':   pred,
            'risk_prob':    round(risk_prob * 100, 1),
            'risk_level':   risk_level,
            'color':        color,
            'weekly_hours': recommend_hours(data, risk_prob),
            'tips':         get_tips(risk_prob, data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("="*50)
    print("  Student Risk Prediction — CSE274 ML Project")
    print("="*50)
    print("  Open: http://127.0.0.1:5000")
    print("  Ctrl+C to stop\n")
    app.run(debug=True, port=5000)
