from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from fastai.vision.all import load_learner, PILImage
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import joblib
import pandas as pd
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from fertilizer_recommendations import get_fertilizer_recommendation
from functools import wraps
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Cache configuration
cache_config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = frozenset({'png', 'jpg', 'jpeg', 'gif', 'webp'})
MODEL_CACHE_TIMEOUT = 3600  # 1 hour

app.config.update(
    SECRET_KEY='your_secret_key_here',
    SQLALCHEMY_DATABASE_URI='sqlite:///users.db',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    UPLOAD_FOLDER=UPLOAD_FOLDER
)

# Initialize database
db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)

class Report(db.Model):
    __tablename__ = 'reports'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Soil parameters
    nitrogen = db.Column(db.Float)
    phosphorus = db.Column(db.Float)
    potassium = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    ph = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    
    # Predictions
    predicted_crop = db.Column(db.String(100))
    crop_confidence = db.Column(db.Float)
    predicted_disease = db.Column(db.String(100))
    disease_confidence = db.Column(db.Float)
    predicted_deficiency = db.Column(db.String(100))
    deficiency_confidence = db.Column(db.Float)
    fertilizer_recommendation = db.Column(db.Text)
    
    # Image path (optional)
    image_path = db.Column(db.String(255))

# Create database tables
with app.app_context():
    db.create_all()

# Model loading with caching
@cache.memoize(timeout=MODEL_CACHE_TIMEOUT)
def load_models():
    """Load and cache all models"""
    try:
        return {
            'learn_inf': load_learner('models/EfficientNetB0_best.pkl'),
            'crop_model': joblib.load('models/RandomForest_label.pkl'),
            'deficiency_model': joblib.load('models/RandomForest_deficiency.pkl'),
            'scaler': joblib.load('models/scaler.pkl')
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Initialize models on startup
models = load_models()

# Utility functions
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@cache.memoize(timeout=300)
def predict_image(img_path):
    """Cached image prediction"""
    try:
        img = PILImage.create(img_path)
        pred_class, pred_idx, outputs = models['learn_inf'].predict(img)
        return pred_class, outputs.max().item()
    except Exception as e:
        logger.error(f"Error predicting image: {str(e)}")
        return None, None

def predict_crop_and_deficiency(soil_data):
    """Optimized crop and deficiency prediction"""
    try:
        scaled_data = models['scaler'].transform(
            soil_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        )
        soil_data_scaled = pd.DataFrame(
            scaled_data,
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )
        
        crop_prediction = models['crop_model'].predict(soil_data_scaled)
        crop_proba = models['crop_model'].predict_proba(soil_data_scaled).max()
        deficiency_prediction = models['deficiency_model'].predict(soil_data_scaled)
        deficiency_proba = models['deficiency_model'].predict_proba(soil_data_scaled).max()

        N, P, K, pH = soil_data[['N', 'P', 'K', 'ph']].values[0]
        fertilizer_recommendation = get_fertilizer_recommendation(
            N, P, K, pH, crop=crop_prediction[0]
        )

        return (
            crop_prediction[0],
            crop_proba,
            deficiency_prediction[0],
            deficiency_proba,
            fertilizer_recommendation
        )
    except Exception as e:
        logger.error(f"Error predicting crop and deficiency: {str(e)}")
        return None, None, None, None, None

def save_report(soil_data, predictions, image_path=None):
    try:
        report = Report(
            user_id=session['user_id'],
            nitrogen=soil_data.get('N'),
            phosphorus=soil_data.get('P'),
            potassium=soil_data.get('K'),
            temperature=soil_data.get('temperature'),
            humidity=soil_data.get('humidity'),
            ph=soil_data.get('ph'),
            rainfall=soil_data.get('rainfall'),
            predicted_crop=predictions.get('crop'),
            crop_confidence=predictions.get('crop_confidence'),
            predicted_disease=predictions.get('disease_class'),
            disease_confidence=predictions.get('disease_confidence'),
            predicted_deficiency=predictions.get('predicted_deficiency'),
            deficiency_confidence=predictions.get('deficiency_confidence'),
            fertilizer_recommendation=predictions.get('fertilizer_recommendation'),
            image_path=image_path
        )
        db.session.add(report)
        db.session.commit()
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        db.session.rollback()

# Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        if not all(data.get(field) for field in ['username', 'email', 'password', 'confirm_password']):
            return render_template('signup.html', error='Invalid input')
        
        if data['password'] != data['confirm_password']:
            return render_template('signup.html', error='Passwords do not match')

        if User.query.filter((User.username == data['username']) | 
                           (User.email == data['email'])).first():
            return render_template('signup.html', error='Username or email already exists')

        try:
            new_user = User(
                username=data['username'],
                email=data['email'],
                password=generate_password_hash(data['password'])
            )
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration failed: {str(e)}")
            return render_template('signup.html', error='Registration failed')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/reports')
@login_required
def view_reports():
    try:
        page = request.args.get('page', 1, type=int)
        reports = Report.query.filter_by(user_id=session['user_id'])\
            .order_by(Report.timestamp.desc())\
            .paginate(page=page, per_page=10)
        return render_template('reports.html', reports=reports)
    except Exception as e:
        logger.error(f"Error viewing reports: {str(e)}")
        flash('An error occurred while loading reports.', 'error')
        return redirect(url_for('index'))
@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    try:
        report = Report.query.get_or_404(report_id)
        if report.user_id != session['user_id']:
            flash('You do not have permission to view this report.', 'error')
            return redirect(url_for('view_reports'))
        return render_template('report_detail.html', report=report)
    except Exception as e:
        logger.error(f"Error viewing report details: {str(e)}")
        flash('An error occurred while loading the report details.', 'error')
        return redirect(url_for('view_reports'))
@app.route('/predict_soil', methods=['POST'])
@login_required
def predict_soil_route():
    try:
        soil_data = {
            key: float(request.form[key]) 
            for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        }
        soil_df = pd.DataFrame([soil_data])
        
        crop, crop_conf, deficiency, def_conf, fertilizer = predict_crop_and_deficiency(soil_df)
        
        predictions = {
            'crop': crop,
            'crop_confidence': round(crop_conf * 100, 2),
            'predicted_deficiency': deficiency,
            'deficiency_confidence': round(def_conf * 100, 2),
            'fertilizer_recommendation': fertilizer
        }
        
        save_report(soil_data, predictions)
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error in predict_soil: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_image', methods=['POST'])
@login_required
def predict_image_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_class, img_confidence = predict_image(filepath)
        if img_class is None:
            return jsonify({'error': 'Error processing image'}), 500

        predictions = {
            'disease_class': img_class,
            'disease_confidence': round(img_confidence * 100, 2)
        }
        
        save_report({}, predictions, image_path=filepath)
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error in predict_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_hybrid', methods=['POST'])
@login_required
def predict_hybrid_route():
    try:
        soil_data = {
            key: float(request.form[key]) 
            for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        }
        soil_df = pd.DataFrame([soil_data])
        
        crop, crop_conf, deficiency, def_conf, fertilizer = predict_crop_and_deficiency(soil_df)
        
        image_path = None
        img_class = img_confidence = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                img_class, img_confidence = predict_image(image_path)

        predictions = {
            'hybrid_prediction': f"It is classified as a disease: {img_class}." if img_class else None,
            'crop': crop,
            'crop_confidence': round(crop_conf * 100, 2),
            'predicted_deficiency': deficiency,
            'deficiency_confidence': round(def_conf * 100, 2),
            'fertilizer_recommendation': fertilizer,
            'disease_class': img_class,
            'disease_confidence': round(img_confidence * 100, 2) if img_confidence is not None else None
        }
        
        save_report(soil_data, predictions, image_path=image_path)
        return jsonify(predictions)
    except Exception as e:
        logger.error(f"Error in predict_hybrid: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if __name__ == '__main__':
    app.run(debug=False)