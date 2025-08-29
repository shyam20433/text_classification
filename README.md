# 🌿 Mind Garden Assessment

A comprehensive mental health assessment tool that uses machine learning models and natural language processing to analyze psychological responses and detect signs of depression.

## 📋 Features

- **Interactive Survey**: 15 carefully crafted questions covering emotional, behavioral, and scenario-based responses
- **AI-Powered Analysis**: Uses Cohere's AI to provide psychological insights and mental health condition analysis
- **Multi-Model ML Prediction**: Leverages 6 different machine learning models for accurate depression detection:
  - Decision Tree
  - Random Forest
  - XGBoost
  - Logistic Regression
  - LightGBM
  - TensorFlow Neural Network
- **Professional UI**: Modern, responsive design with smooth animations and mobile-friendly interface
- **Data Storage**: MongoDB integration for storing assessment results
- **Session Management**: Secure session-based result handling

## 🚀 Installation

### Prerequisites

- Python 3.8+
- MongoDB (running locally on port 27017)
- Cohere API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd ml-projects
   ```

2. **Install dependencies**
   ```bash
   pip install flask cohere joblib nltk numpy tensorflow pymongo
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Configure environment**
   - Update the Cohere API key in `app.py`
   - Ensure MongoDB is running on localhost:27017
   - Set a strong secret key for Flask sessions

5. **Organize model files**
   ```
   models/
   ├── decision_tree.pkl
   ├── random_forest.pkl
   ├── xgboost.pkl
   ├── logistic_regression.pkl
   ├── lightgbm.pkl
   ├── tf_model.keras
   ├── tfidf.pkl
   └── label_encoder.pkl
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - Complete the 15-question assessment
   - View your personalized psychological analysis and depression risk assessment

3. **Understanding Results**
   - **Psychological Summary**: AI-generated analysis of your emotional state
   - **Depression Result**: Clear classification (Depressed/Not Depressed)
   - **Model Predictions**: Detailed outputs from all 6 ML models (toggleable)

## 🏗️ Project Structure

```
ml-projects/
├── app.py                 # Main Flask application
├── models/               # Machine learning models
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── logistic_regression.pkl
│   ├── lightgbm.pkl
│   ├── tf_model.keras
│   ├── tfidf.pkl
│   └── label_encoder.pkl
├── templates/            # HTML templates
│   ├── index.html        # Main assessment page
│   └── results.html      # Results display page
├── Suicide_Detection.csv # Training dataset
├── depression_detector.py # Model training script
├── text.py              # Text processing utilities
└── README.md            # This file
```

## 🔧 Technical Details

### Machine Learning Models
- **Text Preprocessing**: NLTK tokenization, stopword removal, and Porter stemming
- **Feature Extraction**: TF-IDF vectorization
- **Model Ensemble**: Majority voting system for final prediction
- **Model Performance**: Each model is trained on suicide detection dataset

### API Integration
- **Cohere AI**: For psychological analysis and text summarization
- **MongoDB**: For storing assessment results and user data

### Frontend Technologies
- **HTML5/CSS3**: Modern, responsive design
- **JavaScript**: Interactive survey functionality
- **Font Awesome**: Icons for enhanced UI
- **CSS Gradients**: Professional visual effects

## 📊 Assessment Questions

The application includes 15 questions across three categories:

### Word Association (3 questions)
- Alone, Failure, Joy

### Scenario-Based (7 questions)
- Ideal day description, morning feelings, given-up goals, emotional weather, comfort sources, hidden thoughts, "okay" definition

### Behavioral (5 questions)
- Energy levels, social withdrawal, hobby enjoyment, stress response, support network

## 🔒 Privacy & Security

- Session-based data handling
- No persistent user identification
- Secure API key management
- Local MongoDB storage

## 🚨 Important Notes

- This tool is for educational and screening purposes only
- Not a substitute for professional mental health diagnosis
- Always consult healthcare professionals for clinical assessment
- Results are based on ML models and should be interpreted carefully

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational purposes. Please ensure compliance with local regulations regarding mental health applications.

## 🆘 Support

For technical issues or questions:
- Check the MongoDB connection
- Verify all model files are in the `models/` directory
- Ensure Cohere API key is valid
- Check Python dependencies are installed

---

**Disclaimer**: This application is designed for educational and screening purposes only. It should not replace professional mental health evaluation or treatment. 