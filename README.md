# 🔐 AI-based Adaptive Authentication

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Components](#components)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Models](#models)
- [Constraints](#constraints)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Project Overview

This project implements an AI-based adaptive authentication system using various machine learning models for anomaly detection in user behavior. The system aims to enhance security by identifying unusual patterns in user login activities and authentication attempts.

## 🧩 Components

The project consists of several interrelated components:

1. **User Profiling (`user_profiling.py`)**: 
   - Creates comprehensive profiles for users based on their behavior data.
   - Analyzes login times, device usage, session durations, and action denials.

2. **Data Preprocessing (`preprocess_data.py`)**: 
   - Handles the preprocessing of raw log data.
   - Performs data cleaning, feature extraction, and session analysis.

3. **Anomaly Detection Models**:
   - **Isolation Forest (`isolation_forest_model.py`)**: Implements the Isolation Forest algorithm.
   - **Local Outlier Factor (`lof_model.py`)**: Implements the LOF algorithm.
   - **One-Class SVM (`ocsvm_model.py`)**: Implements the One-Class SVM algorithm.

4. **Model Comparison (`model_comparator.py`)**: 
   - Compares the performance of different anomaly detection models.
   - Provides insights into model effectiveness and feature importance.

5. **Visualization (`anomaly_visualizer.py`)**: 
   - Offers various visualization methods for anomaly detection results.
   - Helps in interpreting model outputs and user behavior patterns.

## 🛠 Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **Concurrent.futures**: Parallel processing

## ✨ Features

- 👤 User profiling based on historical behavior
- 🕵️ Anomaly detection using multiple machine learning models
- 📊 Comprehensive model comparison and evaluation
- 📈 Visualization of anomalies and user patterns
- 🔄 Adaptive learning from user behavior
- 🚀 Parallel processing for efficient model training

## 🤖 Models

This project implements and compares three anomaly detection models:

1. **Isolation Forest**: 
   - Effective for high-dimensional datasets
   - Isolates anomalies instead of profiling normal points

2. **Local Outlier Factor (LOF)**:
   - Identifies anomalous data points by measuring the local deviation of a given data point with respect to its neighbors

3. **One-Class SVM (OCSVM)**:
   - Learns a decision function for novelty detection
   - Classifies new data as similar or different to the training set

## 🚧 Constraints

- Requires sufficient historical data for accurate user profiling
- Performance may vary depending on the quality and quantity of input data
- May require periodic retraining to adapt to evolving user behaviors
- Privacy considerations must be taken into account when handling user data

## 📥 Installation

```bash
git clone https://github.com/your-username/ai-based-adaptive-authentication.git
cd ai-based-adaptive-authentication
pip install -r requirements.txt
```
## 🚀 Usage

1. **Prepare your data**:
```javascript
from preprocess_data import Preprocessor

preprocessor = Preprocessor('path/to/your/data.csv')
preprocessed_data = preprocessor.preprocess()
```

2. **Prepare your data**:
```javascript
from isolation_forest_model import IsolationForestModel
from lof_model import LOFModel
from ocsvm_model import OCSVMModel
from model_comparator import ModelComparator

iforest_model = IsolationForestModel(preprocessed_data)
lof_model = LOFModel(preprocessed_data)
ocsvm_model = OCSVMModel(preprocessed_data)

comparator = ModelComparator(lof_model, iforest_model, ocsvm_model)
comparator.run_comparison()
```

3. **Analyze individual users**:
```javascript
# Example for Isolation Forest model
example_user_id = 'user123'
iforest_model.analyze_user(example_user_id, print_results=True)

# Visualize anomalies for a specific user
iforest_model.visualize_anomalies(example_user_id)
```

4. **Visualize the results**:
```javascript
from anomaly_visualizer import AnomalyVisualizer

visualizer = AnomalyVisualizer()
visualizer.visualize_user_anomalies(user_id, user_data, features)
visualizer.visualize_feature_distributions(user_id, user_data)
```

5. **Make predictions for new actions**:
```javascript
example_action_features = {
    'hour_of_timestamp': 15,
    'phone_versions': 'iPhone14_5',
    'iOS sum': 1,
    'Android sum': 0,
    'is_denied': 0,
    'session_duration': 300,
    'location_or_ip': 'New York, USA (192.168.1.1)'
}
prediction = iforest_model.predict_user_action(example_user_id, example_action_features)
print(f"Prediction for new action: {prediction}")
```

## 👥 Contributing
Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
This README provides a comprehensive overview of your project, including its components, features, technologies used, and instructions for installation and usage. The use of emojis and markdown formatting enhances the visual appeal and readability of the document. You can further customize this README by adding specific details about your implementation, project goals, or any other relevant information.
```