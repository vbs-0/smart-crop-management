from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, session
from flask_cors import CORS
from fastai.vision.all import load_learner, PILImage
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import os
import logging
import traceback
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from fertilizer_recommendations import get_fertilizer_recommendation

app = Flask(__name__)
CORS(app)



# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Configuration
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize extensions
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()



# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
try:
    learn_inf = load_learner('models/EfficientNetB0_best.pkl')
    crop_model = joblib.load('models/RandomForest_label.pkl')
    deficiency_model = joblib.load('models/RandomForest_deficiency.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    try:
        img = PILImage.create(img_path)
        pred_class, pred_idx, outputs = learn_inf.predict(img)
        return pred_class, outputs.max().item()
    except Exception as e:
        logger.error(f"Error predicting image: {str(e)}")
        return None, None


def predict_crop_and_deficiency(soil_data):
    try:
        # Scale the soil data and convert to DataFrame with original feature names
        soil_data_scaled = pd.DataFrame(scaler.transform(soil_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]),
                                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Make predictions for crop and deficiency
        crop_prediction = crop_model.predict(soil_data_scaled)
        crop_proba = crop_model.predict_proba(soil_data_scaled).max()
        deficiency_prediction = deficiency_model.predict(soil_data_scaled)
        deficiency_proba = deficiency_model.predict_proba(soil_data_scaled).max()

        # Get fertilizer recommendation
        N, P, K, pH = soil_data[['N', 'P', 'K', 'ph']].values[0]
        fertilizer_recommendation = get_fertilizer_recommendation(N, P, K, pH, crop=crop_prediction[0])

        return crop_prediction[0], crop_proba, deficiency_prediction[0], deficiency_proba, fertilizer_recommendation
    except Exception as e:
        logger.error(f"Error predicting crop and deficiency: {str(e)}")
        return None, None, None, None, None






@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if not username or not email or not password or password != confirm_password:
            return render_template('signup.html', error='Invalid input')

        # Check if user already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            return render_template('signup.html', error='Username or email already exists')

        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            return render_template('signup.html', error='Registration failed')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Find user
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Add @login_required decorator and check to existing routes
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Modify existing routes to use login_required
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



@app.route('/predict_soil', methods=['POST'])
def predict_soil_route():
    try:
        logger.debug("Received data: %s", request.form)

        # Process soil data
        soil_data = {key: float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']}
        soil_df = pd.DataFrame([soil_data])
        logger.info("Processed soil data: %s", soil_data)
        
        crop, crop_conf, deficiency, def_conf, fertilizer = predict_crop_and_deficiency(soil_df)

        return jsonify({
            'crop': crop,
            'crop_confidence': round(crop_conf * 100, 2),
            'predicted_deficiency': deficiency,
            'deficiency_confidence': round(def_conf * 100, 2),
            'fertilizer_recommendation': fertilizer
        })

    except KeyError as e:
        logger.error("Missing required field: %s", str(e), exc_info=True)
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please check the logs for more details.'}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    try:
        logger.debug("Received files: %s", request.files)

        # Process image data
        img_class, img_confidence = None, None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                img_class, img_confidence = predict_image(filepath)
                logger.info("Image prediction: class=%s, confidence=%s", img_class, img_confidence)

        if img_class is None:
            return jsonify({'error': 'Error processing image'}), 500

        return jsonify({
            'disease_class': img_class,
            'disease_confidence': round(img_confidence * 100, 2)
        })

    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please check the logs for more details.'}), 500

@app.route('/predict_hybrid', methods=['POST'])
def predict_hybrid_route():
    try:
        logger.debug("Received data: %s", request.form)
        logger.debug("Received files: %s", request.files)

        # Process soil data
        soil_data = {key: float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']}
        soil_df = pd.DataFrame([soil_data])
        logger.info("Processed soil data: %s", soil_data)
        
        crop, crop_conf, deficiency, def_conf, fertilizer = predict_crop_and_deficiency(soil_df)
        
        # Process image data
        img_class, img_confidence = None, None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                img_class, img_confidence = predict_image(filepath)
                logger.info("Image prediction: class=%s, confidence=%s", img_class, img_confidence)

        if crop is None and img_class is None:
            return jsonify({'error': 'Error processing data'}), 500

        # Generate hybrid prediction
        return jsonify({
    'hybrid_prediction': f"It is classified as a disease: {img_class}.",
    'crop': crop,
    'crop_confidence': round(crop_conf * 100, 2),
    'predicted_deficiency': deficiency,
    'deficiency_confidence': round(def_conf * 100, 2),
    'fertilizer_recommendation': fertilizer,
    'disease_class': img_class,
    'disease_confidence': round(img_confidence * 100, 2) if img_confidence is not None else None
})


    except KeyError as e:
        logger.error("Missing required field: %s", str(e), exc_info=True)
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        logger.error("Invalid value: %s", str(e), exc_info=True)
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please check the logs for more details.'}), 500

if __name__ == '__main__':
    app.run(debug=False)
