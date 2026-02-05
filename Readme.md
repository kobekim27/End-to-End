# End-to-End Income Prediction System

A comprehensive machine learning project that builds, trains, and deploys a predictive model to classify income levels using the Adult dataset. This project demonstrates the complete data science workflow from data ingestion and preprocessing to model training and API deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [API Documentation](#api-documentation)
- [Results](#results)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an end-to-end machine learning pipeline for income prediction. It processes the Adult dataset through multiple stages:

1. **Data Loading & Normalization**: CSV to relational database conversion with normalization
2. **Exploratory Data Analysis**: Statistical analysis and visualization of features
3. **Data Preprocessing**: Feature engineering, encoding, and scaling
4. **Model Training**: Training multiple classifiers with hyperparameter tuning
5. **Model Evaluation**: Cross-validation and performance metrics
6. **API Deployment**: RESTful API for real-time predictions

## Dataset

**Adult Dataset** (also known as Census Income Dataset)
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
- **Size**: ~32,561 records with 14 features
- **Target Variable**: Income (binary: >50K or ≤50K)
- **Features**:
  - Demographics: age, sex, race, native-country
  - Education: education, education-num
  - Employment: workclass, occupation, hours-per-week
  - Financial: capital-gain, capital-loss
  - Other: fnlwgt, marital-status, relationship

## Project Structure

```
End-to-End/
├── Readme.md                          # This file
├── EAS503 Final/
│   ├── main.py                        # FastAPI application for predictions
│   ├── End-to-End.ipynb               # Jupyter notebook with full pipeline
│   ├── final_model.joblib             # Trained ML model
│   ├── adult-all.csv                  # Adult dataset
│   └── requirements.txt                # Python dependencies
├── jupyter_book/                      # Jupyter Book configuration
├── mynewbook/                         # Built documentation
├── stramlit/                          # Streamlit application (optional)
└── mlruns/                            # MLflow experiment tracking

```

## Features

### Data Processing
- ✅ CSV parsing with automated column mapping
- ✅ SQLite database normalization
- ✅ Handling missing values and data validation
- ✅ Feature scaling and encoding

### Machine Learning
- ✅ Multiple classifier implementations
- ✅ Cross-validation (k-fold)
- ✅ Hyperparameter tuning
- ✅ Feature importance analysis
- ✅ Model serialization with joblib

### API & Deployment
- ✅ RESTful API with FastAPI
- ✅ Real-time prediction endpoint
- ✅ Request validation with Pydantic
- ✅ CORS support for cross-origin requests

### Experiment Tracking
- ✅ MLflow integration for experiment logging
- ✅ Metrics and parameter tracking
- ✅ Model versioning

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd End-to-End
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd "EAS503 Final"
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Full Pipeline (Jupyter Notebook)

```bash
jupyter notebook "End-to-End.ipynb"
```

The notebook executes the complete workflow:
- Data loading and normalization
- Exploratory data analysis
- Preprocessing and feature engineering
- Model training and evaluation
- Results visualization

### 2. Start the API Server

```bash
cd "EAS503 Final"
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### 3. Make Predictions via API

**Health Check:**
```bash
curl http://localhost:8000/
```

**Make a Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [39, 7, 77516, 13, 13, 4, 8, 3, 2, 1, 0, 0, 40, 39, 0]}'
```

## Model Architecture

### Training Pipeline

```
Raw Data (CSV)
    ↓
Data Loading & Parsing
    ↓
SQLite Database Normalization
    ↓
Exploratory Data Analysis
    ↓
Preprocessing (Missing Values, Encoding, Scaling)
    ↓
Feature Engineering
    ↓
Train-Test Split (80-20)
    ↓
Model Selection & Training
    ↓
Cross-Validation (5-fold)
    ↓
Hyperparameter Tuning
    ↓
Final Model Evaluation
    ↓
Model Serialization (joblib)
```

### Supported Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Neural Networks (if applicable)

## API Documentation

### Endpoints

#### GET `/`
**Description**: Health check endpoint

**Response**:
```json
{
  "message": "Income Prediction API"
}
```

#### POST `/predict`
**Description**: Make income prediction for given features

**Request Body**:
```json
{
  "features": [
    age, workclass_id, fnlwgt, education_id, education_num,
    marital_status_id, occupation_id, relationship_id, race_id,
    sex_id, capital_gain, capital_loss, hours_per_week,
    native_country_id
  ]
}
```

**Response** (Success):
```json
{
  "prediction": 0  // 0: Income ≤50K, 1: Income >50K
}
```

**Response** (Error):
```json
{
  "detail": "Error message"
}
```

### Feature Ordering
Features must be provided in the following order:
1. age (integer)
2. workclass_id (integer)
3. fnlwgt (integer)
4. education_id (integer)
5. education_num (integer)
6. marital_status_id (integer)
7. occupation_id (integer)
8. relationship_id (integer)
9. race_id (integer)
10. sex_id (integer)
11. capital_gain (integer)
12. capital_loss (integer)
13. hours_per_week (integer)
14. native_country_id (integer)

## Results

The trained model (`final_model.joblib`) achieves:
- **Accuracy**: [See notebook for actual metrics]
- **Precision**: [See notebook for actual metrics]
- **Recall**: [See notebook for actual metrics]
- **F1-Score**: [See notebook for actual metrics]

Detailed evaluation metrics and visualizations are available in the Jupyter notebook.

## Development

### Directory Layout

- **EAS503 Final/**: Main project directory
  - `End-to-End.ipynb`: Complete pipeline implementation
  - `main.py`: API server code
  - `final_model.joblib`: Serialized trained model
  - `adult-all.csv`: Dataset

- **jupyter_book/**: Jupyter Book configuration for documentation
- **mynewbook/**: Generated HTML documentation
- **mlruns/**: MLflow experiment tracking data

### Dependencies

| Package | Purpose |
|---------|---------|
| fastapi | Web framework for API |
| uvicorn | ASGI server |
| scikit-learn | Machine learning library |
| joblib | Model serialization |
| numpy | Numerical computing |
| pandas | Data manipulation |
| matplotlib | Visualization |
| seaborn | Statistical visualization |

See `requirements.txt` for complete list and versions.

### Running Tests

```bash
# Run unit tests (if available)
pytest tests/

# Run API tests
pytest tests/test_api.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Performance Considerations

- Model is loaded once at API startup for efficiency
- Predictions are computed using vectorized NumPy operations
- Consider implementing caching for repeated predictions
- For production, use a production-grade ASGI server (Gunicorn + Uvicorn)

## Deployment

### Docker

A Dockerfile is provided for containerization:

```bash
docker build -t income-predictor .
docker run -p 8000:8000 income-predictor
```

### Production Deployment

For production environments:

1. Use environment variables for configuration
2. Implement request logging and monitoring
3. Add authentication/authorization if needed
4. Use a production ASGI server (Gunicorn)
5. Implement rate limiting
6. Add comprehensive error handling
7. Set up monitoring and alerting

## Troubleshooting

### Model Not Found
Ensure `final_model.joblib` is in the same directory as `main.py`

### Dataset Not Found
Ensure `adult-all.csv` is in the correct directory when running the notebook

### Port Already in Use
```bash
uvicorn main:app --reload --port 8001
```

## References

- [Adult Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Adult)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

## License

This project is provided for educational purposes. Please refer to the original dataset license and acknowledgments.

## Acknowledgments

- Dataset source: [Lichman, M. (2013). UCI Machine Learning Repository](https://archive.ics.uci.edu/ml)
- Original dataset creators: Bache, K. & Lichman, M.
- Built for EAS503 Final Project at University at Buffalo

---

**Last Updated**: February 2026

For questions or issues, please open an issue on the repository or contact the maintainers.
