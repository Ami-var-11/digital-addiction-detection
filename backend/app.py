from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'addiction_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# Features expected by the model
FEATURES = ['daily_screen_time_min', 'num_app_switches', 'social_media_time_min']

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'AI Addiction Detection API',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict addiction risk from Android app data
    Expected JSON: {
        "screen_time": 370,
        "app_count": 12,
        "social_time": 145
    }
    """
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        screen_time = float(data.get('screen_time', 0))
        app_count = float(data.get('app_count', 0))
        social_time = float(data.get('social_time', 0))
        
        # Validate
        if screen_time <= 0:
            return jsonify({'error': 'Invalid screen time'}), 400
        
        # Prepare features for model
        features = np.array([[screen_time, app_count, social_time]])
        
        # Make prediction
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0].tolist()
        
        # Calculate risk level
        risk_prob = probabilities[1]
        
        if risk_prob >= 0.7:
            risk_level = "HIGH"
        elif risk_prob >= 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        # Generate recommendations
        recommendations = []
        
        if screen_time > 480:  # 8+ hours
            recommendations.append({
                'priority': 'high',
                'message': f"Screen time ({screen_time/60:.1f}h) exceeds 8 hours. Try reducing by 30 minutes daily."
            })
        
        social_percent = (social_time / screen_time * 100) if screen_time > 0 else 0
        if social_percent > 50:
            recommendations.append({
                'priority': 'high',
                'message': f"Social media is {social_percent:.1f}% of usage. Set app timers."
            })
        
        if app_count > 20:
            recommendations.append({
                'priority': 'medium',
                'message': f"You used {app_count} different apps. Try focusing on one task at a time."
            })
        
        if risk_level == "HIGH":
            recommendations.append({
                'priority': 'high',
                'message': "High addiction risk detected. Consider a digital detox day."
            })
        elif risk_level == "LOW":
            recommendations.append({
                'priority': 'low',
                'message': "Low risk - you're doing well! Keep monitoring."
            })
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_probability': round(risk_prob * 100, 1),
            'probabilities': {
                'normal': round(probabilities[0] * 100, 1),
                'addicted': round(probabilities[1] * 100, 1)
            },
            'recommendations': recommendations,
            'features_used': {
                'screen_time_hours': round(screen_time / 60, 1),
                'app_count': int(app_count),
                'social_percentage': round(social_percent, 1)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {risk_level} risk ({risk_prob:.1%})")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'features': FEATURES,
        'accuracy': 0.81,
        'feature_importance': {
            'social_media_time': 0.44,
            'screen_time': 0.33,
            'app_switches': 0.23
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 AI Addiction Detection API")
    print("="*60)
    print(f"Model loaded: {model is not None}")
    print(f"Features: {FEATURES}")
    print("\nEndpoints:")
    print("  GET  /          - Health check")
    print("  POST /predict   - Make prediction")
    print("  GET  /model_info - Model details")
    print("\nStarting server...")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)