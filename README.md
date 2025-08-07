# ML Prediction API

A FastAPI-based machine learning service that provides predictions for two different models: a Random Forest model for occurrence prediction and an XGBoost model for damage assessment.

## Features

- **Dual Model Support**: Separate endpoints for Random Forest and XGBoost models
- **RESTful API**: Clean REST endpoints built with FastAPI
- **Input Validation**: Pydantic models ensure data integrity
- **Error Handling**: Comprehensive error handling with meaningful HTTP status codes
- **Production Ready**: Includes Gunicorn for production deployment

## API Endpoints

### 1. Occurrence Prediction
**Endpoint**: `POST /predict/happen`

Uses a Random Forest model to predict whether an event will occur.

**Request Body**:
```json
{
  "features": [12.9716, 77.5946, 24.0, 1400, 45.2, 920.5, 15.3, 2.8]
}
```

*Features in order: Latitude, Longitude, Duration, Time, Rainfall, Elevation, Slope, Distance*

**Response**:
```json
{
  "prediction": [0]
}
```

### 2. Damage Assessment
**Endpoint**: `POST /predict/damage`

Uses an XGBoost model to predict damage levels or severity.

**Request Body**:
```json
{
  "features": [12.9716, 77.5946, 24.0, 1400, 45.2, 920.5, 15.3, 2.8]
}
```

*Features in order: Latitude, Longitude, Duration, Time, Rainfall, Elevation, Slope, Distance*

**Response**:
```json
{
  "prediction": [2]  "high risk"
}
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**:
   - `Random_forest_model.pkl` - Random Forest model for occurrence prediction
   - `XGBOOST.pkl` - XGBoost model for damage assessment
   
   Place these files in the root directory alongside `main.py`.

## Running the Application

### Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative API docs**: `http://localhost:8000/redoc` (ReDoc)

## Usage Examples

### Using curl

**Occurrence Prediction**:
```bash
curl -X POST "http://localhost:8000/predict/happen" \
     -H "Content-Type: application/json" \
     -d '{"features": [12.9716, 77.5946, 24.0, 1400, 45.2, 920.5, 15.3, 2.8]}'
```

**Damage Assessment**:
```bash
curl -X POST "http://localhost:8000/predict/damage" \
     -H "Content-Type: application/json" \
     -d '{"features": [12.9716, 77.5946, 24.0, 1400, 45.2, 920.5, 15.3, 2.8]}'
```

### Using Python requests

```python
import requests
import json

url = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

# Example data: [Latitude, Longitude, Duration, Time, Rainfall, Elevation, Slope, Distance]
data = {"features": [12.9716, 77.5946, 24.0, 1400, 45.2, 920.5, 15.3, 2.8]}

# Occurrence prediction
response = requests.post(f"{url}/predict/happen", 
                        headers=headers, 
                        data=json.dumps(data))
print("Occurrence prediction:", response.json())

# Damage assessment
response = requests.post(f"{url}/predict/damage", 
                        headers=headers, 
                        data=json.dumps(data))
print("Damage assessment:", response.json())
```

## Dependencies

- **FastAPI** (0.112.0) - Modern web framework for building APIs
- **Uvicorn** (0.30.0) - ASGI web server implementation
- **Scikit-learn** (1.6.1) - Machine learning library
- **NumPy** (1.26.4) - Numerical computing library
- **Pydantic** (2.8.0) - Data validation using Python type annotations
- **Gunicorn** (22.0.0) - Python WSGI HTTP Server for UNIX
- **Joblib** (1.4.2) - Tools for lightweight pipelining in Python
- **XGBoost** (≥2.1.0) - Gradient boosting framework
- **Imbalanced-learn** (≥0.12.0) - Tools for handling imbalanced datasets

## Error Handling

The API returns appropriate HTTP status codes:
- **200**: Successful prediction
- **400**: Bad request (invalid input data or model error)
- **422**: Validation error (invalid request format)

Error responses include detailed messages:
```json
{
  "detail": "Error description here"
}
```

## Model Requirements

### Input Features
The models expect exactly **8 numerical features** in the following order:

1. **Latitude** - Geographic latitude coordinate
2. **Longitude** - Geographic longitude coordinate  
3. **Duration** - Time duration (units depend on your model training)
4. **Time** - Time value (units depend on your model training)
5. **Rainfall** - Rainfall measurement
6. **Elevation** - Elevation above sea level
7. **Slope** - Terrain slope measurement
8. **Distance** - Distance measurement (units depend on your model training)

### Data Requirements
- **Input Format**: List of 8 numerical features in exact order shown above
- **Data Types**: All features must be numeric (int or float)
- **Feature Count**: Must provide exactly 8 features - no more, no less

## Development

### Project Structure
```
.
├── main.py                 # FastAPI application
├── requirements.txt        # Python dependencies
├── Random_forest_model.pkl # Random Forest model file
├── XGBOOST.pkl            # XGBoost model file
└── README.md              # This file
```

### Adding New Models

To add a new model:
1. Save the trained model using joblib
2. Load it in `main.py` 
3. Create a new endpoint following the existing pattern
4. Update this README with the new endpoint documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please [create an issue](link-to-issues) in the repository.
