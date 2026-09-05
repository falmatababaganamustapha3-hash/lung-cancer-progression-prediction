import os
import joblib
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app import db
from app.models import Prediction

main = Blueprint('main', __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'lung_cancer_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'ml_model', 'model_features.pkl')

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)


@main.route('/')
def home():
    return render_template('home.html')


@main.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction = None

    if request.method == 'POST':
        input_values = []
        for feature in feature_names:
            value = request.form.get(feature, 0)
            input_values.append(float(value))

        raw_prediction = model.predict([input_values])[0]
        prediction = max(1, min(3, raw_prediction))
        prediction = round(prediction, 2)

        new_record = Prediction(
            user_id=current_user.id,
            input_data=str(dict(zip(feature_names, input_values))),
            result=prediction
        )
        db.session.add(new_record)
        db.session.commit()

    return render_template('predict.html', feature_names=feature_names, prediction=prediction)


@main.route('/history')
@login_required
def history():
    records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', records=records)


@main.route('/history/reset', methods=['POST'])
@login_required
def reset_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Your prediction history has been cleared.')
    return redirect(url_for('main.history'))