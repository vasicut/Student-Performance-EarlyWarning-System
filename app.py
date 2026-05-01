"""
Student Performance Early-Warning System
Flask Web Application

Loads the trained Random Forest model exported from the Jupyter notebook
and serves an early-warning dashboard for predicting student academic performance.

Usage:
    python app.py

The app expects the following files in the /models directory:
    - best_model.pkl        (trained Random Forest model)
    - label_encoder.pkl     (target label encoder)
    - feature_columns.pkl   (one-hot encoded feature column order)
    - column_options.json   (categorical column dropdown values)
    - metadata.json         (model metrics, fairness results, feature importance)
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ── Load model artefacts ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

with open(os.path.join(MODEL_DIR, 'column_options.json')) as f:
    column_options = json.load(f)

with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
    metadata = json.load(f)

print(f"Model loaded: {metadata['best_model_name']}")
print(f"Test Accuracy: {metadata['test_accuracy']}")
print(f"Features: {len(feature_columns)}")


def prepare_input(form_data):
    """
    Takes raw form data (categorical + numerical fields),
    builds a one-hot encoded DataFrame matching the training feature order,
    and returns it ready for model.predict().
    """
    raw = {
        'gender': form_data.get('gender'),
        'NationalITy': form_data.get('NationalITy'),
        'PlaceofBirth': form_data.get('PlaceofBirth'),
        'StageID': form_data.get('StageID'),
        'GradeID': form_data.get('GradeID'),
        'SectionID': form_data.get('SectionID'),
        'Topic': form_data.get('Topic'),
        'Semester': form_data.get('Semester'),
        'Relation': form_data.get('Relation'),
        'ParentAnsweringSurvey': form_data.get('ParentAnsweringSurvey'),
        'ParentschoolSatisfaction': form_data.get('ParentschoolSatisfaction'),
        'StudentAbsenceDays': form_data.get('StudentAbsenceDays'),
        'raisedhands': int(form_data.get('raisedhands', 0)),
        'VisITedResources': int(form_data.get('VisITedResources', 0)),
        'AnnouncementsView': int(form_data.get('AnnouncementsView', 0)),
        'Discussion': int(form_data.get('Discussion', 0)),
    }

    row_df = pd.DataFrame([raw])

    cat_cols = metadata['categorical_columns']
    X_encoded = pd.get_dummies(row_df, columns=cat_cols, drop_first=False)

    # Align with training columns
    X_aligned = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in X_encoded.columns:
        if col in X_aligned.columns:
            X_aligned[col] = X_encoded[col].values

    return X_aligned


@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html',
                           column_options=column_options,
                           metadata=metadata)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept student data and return prediction."""
    try:
        X_input = prepare_input(request.form)

        prediction_encoded = model.predict(X_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get probability scores if available
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_input)[0]
            for i, cls in enumerate(label_encoder.classes_):
                probabilities[cls] = round(float(proba[i]) * 100, 1)

        # Risk level mapping
        risk_map = {'L': 'High Risk', 'M': 'Medium Risk', 'H': 'Low Risk'}
        risk_level = risk_map.get(prediction_label, 'Unknown')

        # Recommendations based on prediction
        recommendations = []
        if prediction_label == 'L':
            recommendations = [
                'Schedule immediate one-to-one tutorial sessions',
                'Monitor attendance closely — absence above 7 days is a strong risk indicator',
                'Encourage increased engagement with learning resources',
                'Contact parent/guardian to discuss support strategies',
                'Consider assigning a peer mentor for academic support'
            ]
        elif prediction_label == 'M':
            recommendations = [
                'Monitor engagement metrics weekly',
                'Encourage participation in class discussions',
                'Review resource access patterns for potential improvement',
                'Provide targeted study materials for weaker areas'
            ]
        else:
            recommendations = [
                'Student performing well — maintain current engagement',
                'Consider for peer mentoring roles',
                'Encourage participation in advanced activities'
            ]

        result = {
            'prediction': prediction_label,
            'prediction_full': {'H': 'High', 'M': 'Medium', 'L': 'Low'}[prediction_label],
            'risk_level': risk_level,
            'probabilities': probabilities,
            'recommendations': recommendations,
            'input_summary': {
                'gender': request.form.get('gender'),
                'raised_hands': request.form.get('raisedhands'),
                'visited_resources': request.form.get('VisITedResources'),
                'absence': request.form.get('StudentAbsenceDays'),
                'topic': request.form.get('Topic')
            }
        }

        return render_template('result.html', result=result, metadata=metadata)

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access."""
    try:
        data = request.get_json() or request.form
        X_input = prepare_input(data)

        prediction_encoded = model.predict(X_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_input)[0]
            for i, cls in enumerate(label_encoder.classes_):
                probabilities[cls] = round(float(proba[i]) * 100, 1)

        return jsonify({
            'status': 'success',
            'prediction': prediction_label,
            'prediction_full': {'H': 'High', 'M': 'Medium', 'L': 'Low'}[prediction_label],
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/dashboard')
def dashboard():
    """Render the analytics dashboard."""
    return render_template('dashboard.html', metadata=metadata)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
