# Smart Crop Management 🌱

Welcome to the Smart Crop Management project! This project aims to provide recommendations for crop management based on various parameters.

## Features ✨
- Crop recommendations based on soil health
- Fertilizer recommendations for optimal growth
- User-friendly interface for easy navigation
- Disease prediction from leaf images

## Installation 🛠️
To install the required packages, run:
```
pip install -r requirements.txt
```

## Usage 🚀
1. Clone the repository:
```
git clone https://github.com/vbs-0/smart-crop-management.git
```
2. Navigate to the project directory:
```
cd smart-crop-management
```
3. Run the application:
```
python app_testing.py
```

## File Descriptions 📁
- **app_testing.py**: Main application logic for the Smart Crop Management system.
- **fertilizer_recommendations.py**: Provides fertilizer recommendations based on soil nutrient levels.
- **optamized_app.py**: Contains the Flask application setup, user authentication, and prediction routes.
- **models/**: Contains trained machine learning models for crop prediction and disease detection.
- **templates/**: HTML files for the user interface, including:
  - `index.html`: Main dashboard for predictions.
  - `how_to_use.html`: Guide on how to use the application.
  - `about.html`: Information about the project and team.
- **static/**: Contains static files like images used in the application.
- **crop_recommendation.csv**: Data used for crop recommendations.

## Additional Information ℹ️
- **Database Type**: SQLite, stored in `users.db` file in the project root.
- **Upload Storage**: Uploaded images are stored in the `uploads` directory.
- **Technologies Used**: 
  - Flask: Web framework for building the application.
  - FastAI: Library for deep learning and image classification.
  - SQLAlchemy: ORM for database interactions.
  - Bootstrap: Frontend framework for responsive design.
- **Model Storage**: Trained models are stored in the `models/` directory in `.pkl` format.
- **File Structure Overview**: The project is organized into directories for models, templates, static files, and source code, facilitating easy navigation and maintenance.

## Contributing 🤝
If you would like to contribute to this project, please fork the repository and submit a pull request.

## Created by 👤
VBS
