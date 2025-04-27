import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS  # Import CORS
import joblib
from groq import Groq
import json
import math
from dotenv import load_dotenv
import time
import re

# Load environment variables from .env.local file
load_dotenv('.env.local')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
ENV = os.getenv('FLASK_ENV', 'development')
DEBUG = ENV == 'development'
HOST = '127.0.0.1' if ENV == 'development' else '0.0.0.0'
PORT = int(os.getenv('PORT', 5001))

# Initialize Flask app with additional security headers
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Enable CORS for Next.js frontend
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; font-src 'self' cdnjs.cloudflare.com"
    return response

def load_data(file_path):
    try:
        # Use encoding='utf-8' explicitly for better cross-platform compatibility
        return pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return None

# Load data at startup
DATA_DF = load_data("data/processed_features.csv")
if DATA_DF is not None:
    DATA_DF.columns = DATA_DF.columns.str.strip()
    DATA_DF['patient_id'] = DATA_DF['patient_id'].astype(str)
    logger.info(f"Data loaded successfully at startup. Shape: {DATA_DF.shape}")
    logger.info(f"Columns in processed_features.csv: {DATA_DF.columns.tolist()}")
else:
    logger.error("Error loading processed features data")
    raise Exception("Error loading processed features data")

# Initialize Groq client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.warning(f"Error initializing Groq client: {str(e)}")
        groq_client = None
else:
    logger.warning("GROQ_API_KEY not found in environment variables")
    groq_client = None

# Add these constants at the top of the file
FEATURE_RANGES = {
    'systolic': {
        'min': 85,
        'max': 200,
        'precision': 1,
        'default': 120
    },
    'diastolic': {
        'min': 55,
        'max': 120,
        'precision': 1,
        'default': 80
    },
    'heart_rate': {
        'min': 45,
        'max': 180,
        'precision': 1,
        'default': 75
    },
    'murmur_grade': {
        'min': 1,
        'max': 6,
        'precision': 0,
        'default': 1
    }
}

def normalize_value(value, feature_name):
    """Normalize a value to a reasonable physiological range with appropriate precision"""
    try:
        value = float(value)
        if feature_name in FEATURE_RANGES:
            range_info = FEATURE_RANGES[feature_name]
            
            # If value is clearly invalid, use the default value
            if value < -1000 or value > 1000 or math.isnan(value):
                value = range_info['default']
            
            # Clamp the value to the defined range
            value = min(range_info['max'], max(range_info['min'], value))
            
            # Round to the specified precision
            return round(value, range_info['precision'])
        return round(float(value), 2)
    except (ValueError, TypeError):
        if feature_name in FEATURE_RANGES:
            return FEATURE_RANGES[feature_name]['default']
        return 0.0

def analyze_medical_data(vital_stats, risk_score):
    """Use Groq to analyze medical data and provide heart-disease specific insights"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if API key is properly set
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.error("Groq API key not configured")
                return get_default_analysis(vital_stats, risk_score)

            # Create Groq client
            client = Groq(api_key=api_key)

            # Get patient age and risk level
            age = vital_stats['Patient_Demographics']['Age']
            risk_level = "High Risk" if risk_score > 0.5 else "Low Risk"

            # Prepare the prompt with patient details and vital statistics
            prompt = f"""As a cardiologist, analyze these patient details and vital statistics for heart disease risk factors and provide detailed insights:

Patient Details:
- Age: {age}
- Sex: {vital_stats['Patient_Demographics']['Sex']}
- Height: {vital_stats['Patient_Demographics']['Height']} cm
- Weight: {vital_stats['Patient_Demographics']['Weight']} kg
- Pregnancy Status: {vital_stats['Patient_Demographics']['Pregnancy_Status']}
- Risk Score: {risk_score:.2f} ({risk_level})

Murmur Assessment:
- Presence: {'Present' if vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Presence'] else 'Absent'}
- Location: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Location']}
- Most Audible Location: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Most_Audible_Location']}

Systolic Murmur Details:
- Timing: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Systolic_Details']['Timing']}
- Shape: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Systolic_Details']['Shape']}
- Grade: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Systolic_Details']['Grade']}
- Pitch: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Systolic_Details']['Pitch']}
- Quality: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Systolic_Details']['Quality']}

Diastolic Murmur Details:
- Timing: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Diastolic_Details']['Timing']}
- Shape: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Diastolic_Details']['Shape']}
- Grade: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Diastolic_Details']['Grade']}
- Pitch: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Diastolic_Details']['Pitch']}
- Quality: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']['Diastolic_Details']['Quality']}

Please provide a comprehensive analysis including:
1. Detailed age-specific risk factors and considerations, with specific medical explanations
2. Thorough risk level interpretation based on age and overall health
3. Multiple evidence-based lifestyle recommendations appropriate for the patient's age, with rationale
4. Urgent warning signs (if any) with explanation of their clinical significance
5. Positive health indicators with detailed explanation of their protective effects
6. Multiple age-appropriate follow-up recommendations with timeline and reasoning
7. Multiple potential diagnostic tests to consider based on age and risk level, including rationale and expected findings

Format the response as a structured JSON with these sections:
- risk_factors: {{
    "age_specific_factors": [list of factors with detailed explanations],
    "general_risk_factors": [list of general heart disease risk factors relevant to this patient]
  }}
- cardiovascular_health: {{
    "murmur_status": "detailed explanation of murmur significance",
    "murmur_location": "clinical significance of the location",
    "overall_assessment": "detailed assessment of current cardiovascular status"
  }}
- lifestyle_recommendations: {{
    "age_appropriate_guidance": [list of 3-5 specific recommendations with explanations],
    "diet_recommendations": [specific dietary guidelines],
    "physical_activity": [activity recommendations with frequency and intensity],
    "monitoring_suggestions": [self-monitoring recommendations]
  }}
- warning_signs: {{
    "immediate_actions": [symptoms requiring immediate medical attention],
    "monitoring": [symptoms to monitor closely with explanation]
  }}
- positive_indicators: [at least 3 positive health indicators with detailed explanations]
- follow_up_recommendations: {{
    "immediate_actions": [urgent next steps if needed],
    "monitoring": [regular monitoring recommendations],
    "specialist_follow_up": [specialist referrals if needed with timeline]
  }}
- diagnostic_tests: {{
    "age_appropriate_tests": [list of appropriate tests with rationale],
    "priority_tests": [high-priority tests to consider first],
    "specialized_tests": [advanced diagnostic procedures if indicated]
  }}

Ensure all recommendations are evidence-based and specifically tailored to the patient's age group, risk level, and presence of heart murmur.
"""

            # Try with primary model first
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cardiologist specializing in heart disease risk assessment and prevention. Provide detailed, evidence-based analysis and recommendations that are specifically tailored to the patient's age group. For pediatric patients, focus on growth and development considerations. For adults, focus on age-appropriate lifestyle modifications and monitoring."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="qwen-qwq-32b",  # Using the correct model name
                    temperature=0.3,
                    max_tokens=4096
                )
            except Exception as e:
                logger.error(f"Error with Groq API: {str(e)}")
                return get_default_analysis(vital_stats, risk_score)

            try:
                # Validate response content before attempting to parse
                response_content = chat_completion.choices[0].message.content
                logger.info(f"Received response from Groq (length: {len(response_content) if response_content else 0})")
                
                # Sanitize the response to ensure it's valid JSON
                if not response_content or not response_content.strip():
                    logger.error("Empty response from Groq API")
                    return get_default_analysis(vital_stats, risk_score)
                    
                # Try to find JSON content within the response using pattern matching
                json_match = re.search(r'(\{.*\})', response_content, re.DOTALL)
                if json_match:
                    valid_json = json_match.group(1)
                    try:
                        analysis = json.loads(valid_json)
                        return analysis
                    except json.JSONDecodeError:
                        logger.error("Found JSON-like content but couldn't parse it")
                        # Continue to standard parsing attempt
                
                # Standard parsing attempt
                analysis = json.loads(response_content)
                return analysis
            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                logger.error(f"Error parsing Groq response: {str(e)}")
                # Log the raw response for debugging
                if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
                    logger.error(f"Raw Groq response starts with: {str(chat_completion.choices[0].message.content)[:100]}...")
                if attempt == max_retries - 1:
                    return get_default_analysis(vital_stats, risk_score)
                time.sleep(retry_delay)
                continue

        except Exception as e:
            logger.error(f"Error in Groq analysis (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                return get_default_analysis(vital_stats, risk_score)
            time.sleep(retry_delay)
            continue

    return get_default_analysis(vital_stats, risk_score)

def get_default_analysis(vital_stats, risk_score):
    """Provide a default analysis when the AI service is unavailable"""
    demographics = vital_stats["Patient_Demographics"]
    murmur = vital_stats["Cardiovascular_Metrics"]["Murmur_Assessment"]
    age = demographics['Age']
    
    risk_factors = []
    warnings = []
    
    # Age-specific risk factors
    if age == 'Child':
        risk_factors.append("Pediatric patient - requires specialized cardiac evaluation")
        if murmur["Presence"]:
            risk_factors.append("Heart murmur in pediatric patient - requires careful monitoring of growth and development")
    elif age == 'Adolescent':
        risk_factors.append("Adolescent patient - requires age-appropriate cardiac evaluation")
        if murmur["Presence"]:
            risk_factors.append("Heart murmur in adolescent - requires monitoring of physical development and activity tolerance")
    else:
        risk_factors.append("Adult patient - standard cardiac evaluation protocol")
        if murmur["Presence"]:
            risk_factors.append("Heart murmur in adult patient - requires comprehensive cardiac workup")
    
    # Murmur analysis
    if murmur["Presence"]:
        risk_factors.append(f"Heart murmur detected at {murmur['Location']}")
        if murmur['Systolic_Details']['Grade'] in ['III/VI', 'IV/VI', 'V/VI', 'VI/VI']:
            warnings.append("Significant heart murmur detected - requires immediate evaluation")
    
    # BMI calculation and analysis
    height_m = float(demographics['Height']) / 100  # convert cm to m
    weight_kg = float(demographics['Weight'])
    bmi = weight_kg / (height_m * height_m)
    if bmi > 30:
        risk_factors.append("Obesity (BMI > 30)")
    elif bmi > 25:
        risk_factors.append("Overweight (BMI 25-30)")

    # Age-specific recommendations
    if age == 'Child':
        lifestyle_recommendations = [
            "Maintain a balanced diet appropriate for growth and development",
            "Engage in age-appropriate physical activity (30-60 minutes daily)",
            "Regular monitoring of growth parameters and milestones",
            "Schedule regular pediatric cardiology follow-ups",
            "Ensure proper vaccination schedule",
            "Limit screen time and encourage active play",
            "Maintain regular sleep schedule appropriate for age"
        ]
        follow_up_recommendations = [
            "Schedule pediatric cardiology consultation",
            "Regular growth and development assessment",
            "Monitor for any changes in murmur characteristics",
            "Regular echocardiogram as recommended by pediatric cardiologist",
            "Annual comprehensive pediatric check-up",
            "Regular monitoring of physical activity tolerance"
        ]
        diagnostic_tests = [
            "Pediatric echocardiogram",
            "Growth and development assessment",
            "Basic metabolic panel",
            "Regular cardiac monitoring",
            "Developmental screening",
            "Physical activity tolerance test"
        ]
    elif age == 'Adolescent':
        lifestyle_recommendations = [
            "Maintain a balanced diet with focus on proper nutrition for growth",
            "Engage in regular physical activity (60 minutes daily)",
            "Monitor blood pressure regularly",
            "Maintain healthy weight",
            "Schedule regular cardiac check-ups",
            "Limit processed foods and sugary drinks",
            "Ensure adequate sleep (8-10 hours)",
            "Practice stress management techniques"
        ]
        follow_up_recommendations = [
            "Schedule adolescent cardiology consultation",
            "Regular cardiac monitoring",
            "Annual comprehensive health check-up",
            "Regular blood pressure monitoring",
            "Monitor physical development and activity tolerance",
            "Regular assessment of lifestyle habits"
        ]
        diagnostic_tests = [
            "Echocardiogram",
            "Electrocardiogram (ECG)",
            "Basic metabolic panel",
            "Lipid profile",
            "Physical activity tolerance test",
            "Blood pressure monitoring"
        ]
    else:
        lifestyle_recommendations = [
            "Maintain a heart-healthy diet (Mediterranean or DASH diet)",
            "Engage in regular physical activity (150 minutes weekly)",
            "Monitor blood pressure regularly",
            "Maintain healthy weight",
            "Schedule regular cardiac check-ups",
            "Limit alcohol consumption",
            "Quit smoking if applicable",
            "Manage stress through relaxation techniques"
        ]
        follow_up_recommendations = [
            "Schedule cardiology consultation",
            "Regular cardiac monitoring",
            "Annual comprehensive health check-up",
            "Regular blood pressure monitoring",
            "Regular lipid profile testing",
            "Diabetes screening if indicated"
        ]
        diagnostic_tests = [
            "Echocardiogram",
            "Electrocardiogram (ECG)",
            "Stress test if indicated",
            "Basic metabolic panel",
            "Lipid profile",
            "Coronary calcium score if indicated"
        ]

    return {
        "risk_factors": risk_factors if risk_factors else ["No immediate risk factors identified based on available data"],
        "cardiovascular_health": {
            "murmur_status": "Present" if murmur["Presence"] else "Absent",
            "murmur_location": murmur["Location"],
            "overall_assessment": "Requires attention" if risk_score > 0.5 or murmur["Presence"] else "Generally stable"
        },
        "lifestyle_recommendations": lifestyle_recommendations,
        "warning_signs": warnings if warnings else ["No immediate warning signs detected"],
        "positive_indicators": [
            "Regular monitoring of heart health",
            "Seeking medical guidance for assessment",
            "No heart murmur detected" if not murmur["Presence"] else None
        ],
        "follow_up_recommendations": follow_up_recommendations,
        "diagnostic_tests": diagnostic_tests
    }

# Load Model & Scaler
try:
    try:
        model = tf.keras.models.load_model('heart_disease_model.h5')
        logging.info("Successfully loaded TensorFlow model")
    except Exception as model_error:
        logging.error(f"Error loading TensorFlow model from h5: {str(model_error)}")
        try:
            # Try alternate location
            model = tf.keras.models.load_model('models/heart_disease_model')
            logging.info("Successfully loaded TensorFlow model from alternate location")
        except Exception as alt_error:
            logging.error(f"Error loading TensorFlow model from alternate location: {str(alt_error)}")
            # Create a dummy model for testing
            logging.warning("Creating a dummy model for testing purposes")
            inputs = tf.keras.Input(shape=(36, 1))
            x = tf.keras.layers.Flatten()(inputs)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='binary_crossentropy')
except Exception as e:
    logging.error(f"Error setting up model: {str(e)}")
    model = None

try:
    # Use joblib instead of pickle for better compatibility
    with open('models/scaler.pkl', 'rb') as file:
        scaler = joblib.load(file)
    logging.info("Successfully loaded scaler")
except Exception as e:
    logging.error(f"Error loading scaler: {str(e)}")
    # Create a dummy scaler
    logging.warning("Creating a dummy scaler for testing purposes")
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Configure a dummy transform method
    scaler.transform = lambda x: x  # Just return the input for testing

@app.route('/predict', methods=['GET', 'POST'])
@limiter.limit("30 per minute")
def predict():
    if request.method == 'GET':
        patient_id = request.args.get('patient_id')
    else:
        # Handle JSON data for POST requests
        if request.is_json:
            data = request.get_json()
            patient_id = data.get('patient_id')
        else:
            patient_id = request.form.get('patient_id')
    
    logger.info(f"Received prediction request for patient ID: {patient_id}")

    # Validate Input
    if not patient_id:
        return jsonify({"error": "Missing 'patient_id' in request"}), 400

    # Check if model is available
    if model is None:
        return jsonify({"error": "Model not available. Please check server logs."}), 503

    # Check if scaler is available
    if scaler is None:
        return jsonify({"error": "Scaler not available. Please check server logs."}), 503

    try:
        patient_id = str(int(patient_id))  # Convert safely to int and back to string
    except ValueError:
        return jsonify({"error": "Invalid Patient ID. Must be an integer!"}), 400

    try:
        # First, get patient demographics from training_data.csv
        training_data = pd.read_csv("data/training_data.csv", encoding='utf-8')
        training_data.columns = [col.strip() for col in training_data.columns]
        
        # Print column names for debugging
        logger.info(f"Training data columns: {training_data.columns.tolist()}")
        
        # Ensure required columns exist
        required_columns = ['Patient ID', 'Murmur', 'Murmur locations', 'Most audible location',
                          'Systolic murmur timing', 'Systolic murmur shape', 'Systolic murmur grading',
                          'Systolic murmur pitch', 'Systolic murmur quality', 'Diastolic murmur timing',
                          'Diastolic murmur shape', 'Diastolic murmur grading', 'Diastolic murmur pitch',
                          'Diastolic murmur quality']
        
        missing_columns = [col for col in required_columns if col not in training_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns in training data: {missing_columns}")
            return jsonify({"error": f"Invalid training data format: missing columns {missing_columns}"}), 500
        
        # Convert Patient ID to string and handle any potential whitespace
        training_data['Patient ID'] = training_data['Patient ID'].astype(str).str.strip()
        
        patient_demographics = training_data[training_data['Patient ID'] == patient_id]
        
        if patient_demographics.empty:
            logger.warning(f"Patient ID {patient_id} not found in training data")
            return jsonify({"error": f"Patient demographics not found for ID {patient_id}"}), 404
            
        patient_demographics = patient_demographics.iloc[0]
        
        # Log the raw murmur data for debugging
        logger.info(f"Raw murmur data for patient {patient_id}:")
        logger.info(f"Murmur: {patient_demographics['Murmur']}")
        logger.info(f"Murmur locations: {patient_demographics['Murmur locations']}")
        logger.info(f"Most audible location: {patient_demographics['Most audible location']}")
        logger.info(f"Systolic murmur timing: {patient_demographics['Systolic murmur timing']}")
        
        # Extract demographic details with proper error handling
        try:
            demographics = {
                "Age": str(patient_demographics['Age']),
                "Sex": str(patient_demographics['Sex']),
                "Height": float(patient_demographics['Height']) if pd.notna(patient_demographics['Height']) else 0.0,
                "Weight": float(patient_demographics['Weight']) if pd.notna(patient_demographics['Weight']) else 0.0,
                "Pregnancy Status": str(patient_demographics['Pregnancy status'])
            }
        except KeyError as e:
            logger.error(f"Missing required column in training data: {str(e)}")
            return jsonify({"error": f"Invalid training data format: missing {str(e)}"}), 500

        # Then, get features from processed_features.csv for prediction
        patient_data = DATA_DF[DATA_DF['patient_id'] == patient_id]
        
        if patient_data.empty:
            logger.warning(f"Patient ID {patient_id} not found in processed features dataset")
            return jsonify({"error": f"Patient features not found for ID {patient_id}"}), 404

        # Extract Features (Ensure 36 features)
        X = patient_data.iloc[:, 1:37].values  # Changed from 1:33 to 1:37 to get 36 features
        logger.info(f"Features extracted for patient ID {patient_id}. Shape: {X.shape}")

        # Validate feature count
        if X.shape[1] != 36:  # Changed from 32 to 36
            logger.error(f"Invalid feature count: {X.shape[1]} for patient ID {patient_id}")
            return jsonify({"error": f"Data format error: Expected 36 features, but got {X.shape[1]}"}), 400

        # Normalize Features for prediction
        X_scaled = scaler.transform(X)
        
        # Reshape Correctly
        X_reshaped = X_scaled.reshape(-1, 36, 1)  # Changed from 32 to 36
        
        # Make Prediction
        prediction = model.predict(X_reshaped)[0][0]
        logger.info(f"Prediction made successfully for patient ID {patient_id}: {prediction}")

        # Determine risk level
        risk_level = "High Risk" if prediction > 0.5 else "Low Risk"

        # Process vital statistics with proper normalization
        vital_stats = {
            "Patient_Demographics": {
                "Age": str(patient_demographics['Age']),
                "Sex": str(patient_demographics['Sex']),
                "Height": float(patient_demographics['Height']) if pd.notna(patient_demographics['Height']) else 0.0,
                "Weight": float(patient_demographics['Weight']) if pd.notna(patient_demographics['Weight']) else 0.0,
                "Pregnancy_Status": str(patient_demographics['Pregnancy status'])
            },
            "Cardiovascular_Metrics": {
                "Murmur_Assessment": {
                    "Presence": str(patient_demographics['Murmur']).lower() == 'present',
                    "Location": str(patient_demographics['Murmur locations']).strip() if pd.notna(patient_demographics['Murmur locations']) else 'Single',
                    "Most_Audible_Location": str(patient_demographics['Most audible location']).strip() if pd.notna(patient_demographics['Most audible location']) else 'Aortic',
                    "Diastolic_Details": {
                        "Grade": str(patient_demographics['Diastolic murmur grading']).strip() if pd.notna(patient_demographics['Diastolic murmur grading']) else '4/VI',
                        "Pitch": str(patient_demographics['Diastolic murmur pitch']).strip() if pd.notna(patient_demographics['Diastolic murmur pitch']) else 'High',
                        "Quality": str(patient_demographics['Diastolic murmur quality']).strip() if pd.notna(patient_demographics['Diastolic murmur quality']) else 'Harsh',
                        "Shape": str(patient_demographics['Diastolic murmur shape']).strip() if pd.notna(patient_demographics['Diastolic murmur shape']) else 'Crescendo',
                        "Timing": str(patient_demographics['Diastolic murmur timing']).strip() if pd.notna(patient_demographics['Diastolic murmur timing']) else 'Early'
                    },
                    "Systolic_Details": {
                        "Grade": str(patient_demographics['Systolic murmur grading']).strip() if pd.notna(patient_demographics['Systolic murmur grading']) else '16/VI',
                        "Pitch": str(patient_demographics['Systolic murmur pitch']).strip() if pd.notna(patient_demographics['Systolic murmur pitch']) else 'High',
                        "Quality": str(patient_demographics['Systolic murmur quality']).strip() if pd.notna(patient_demographics['Systolic murmur quality']) else 'Harsh',
                        "Shape": str(patient_demographics['Systolic murmur shape']).strip() if pd.notna(patient_demographics['Systolic murmur shape']) else 'Crescendo',
                        "Timing": str(patient_demographics['Systolic murmur timing']).strip() if pd.notna(patient_demographics['Systolic murmur timing']) else 'Early'
                    }
                }
            }
        }
        
        # Log the murmur details for debugging
        logger.info(f"Murmur details for patient {patient_id}:")
        logger.info(f"Raw murmur data: {vital_stats['Cardiovascular_Metrics']['Murmur_Assessment']}")

        # Get detailed analysis from Groq
        detailed_analysis = analyze_medical_data(vital_stats, prediction)

        # Create detailed response
        response = {
            "patient_id": patient_id,
            "risk_assessment": {
                "risk_level": risk_level,
                "risk_score": float(prediction),
                "interpretation": "High risk of heart disease - immediate medical consultation recommended." if prediction > 0.7
                                else "Moderate to high risk - schedule a cardiac evaluation soon." if prediction > 0.5
                                else "Low to moderate risk - maintain heart-healthy lifestyle." if prediction > 0.3
                                else "Low risk - continue current healthy practices."
            },
            "vital_statistics": vital_stats,
            "detailed_analysis": detailed_analysis,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }

        logger.info(f"Sending response for patient ID {patient_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing request for patient ID {patient_id}: {str(e)}")
        return jsonify({"error": "Error processing request"}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)

