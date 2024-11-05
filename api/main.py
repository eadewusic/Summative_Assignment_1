from enum import Enum
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define model types using Enum
class ModelType(str, Enum):
    LINEAR_REGRESSION = "Linear Regression"
    RANDOM_FOREST = "Random Forest"
    DECISION_TREE = "Decision Tree"
    GRADIENT_DESCENT = "Gradient Descent"

# Pydantic model for input data
class TV(BaseModel):
    tv_spend: float = Field(..., description="TV marketing expenses/spend")
    modelType: ModelType = Field(default=ModelType.LINEAR_REGRESSION, description="Model type")

# Pydantic model for prediction response
class PredictionResponse(BaseModel):
    tv_spend: float
    predicted_sales: float
    modelType: str
    modelRmse: str

# Load the models using joblib
def load_models():
    models = {}
    try:
        # Load the scikit-learn models
        # Updated paths to include 'models/' directory
        models['lr_model'] = joblib.load('models/lr_model.pkl')
        models['dt_model'] = joblib.load('models/dt_model.pkl')
        models['rf_model'] = joblib.load('models/rf_model.pkl')

        # Load Gradient Descent parameters
        # Updated path to include 'models/' directory
        gd_params = joblib.load('models/gd_params.pkl')
        models['gd_model'] = {
            'm': gd_params['m_gd'],
            'b': gd_params['b_gd'],
            'x_mean': gd_params['X_mean'],
            'x_std': gd_params['X_std'],
            'y_mean': gd_params['Y_mean'],
            'y_std': gd_params['Y_std']
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

    return models

# Initialize models
models = load_models()

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: TV):
    """
    Predict sales based on TV marketing spend and selected model type.
    """
    try:
        if data.modelType not in ModelType:
            raise ValueError("Invalid model type specified.")

        # Prepare input for prediction
        tv_spend = np.array([[data.tv_spend]])

        # Get model information based on the selected model type
        if data.modelType == ModelType.GRADIENT_DESCENT:
            # Normalize input for Gradient Descent
            m = models['gd_model']['m']
            b = models['gd_model']['b']
            x_mean = models['gd_model']['x_mean']
            x_std = models['gd_model']['x_std']
            y_mean = models['gd_model']['y_mean']
            y_std = models['gd_model']['y_std']

            # Normalize the input
            tv_spend_normalized = (tv_spend - x_mean) / x_std

            # Make prediction using Gradient Descent
            prediction_normalized = (tv_spend_normalized * m + b)
            prediction = prediction_normalized * y_std + y_mean
            rmse = 'N/A'  # RMSE not applicable for this model
        else:
            # Use scikit-learn models for prediction
            model = models[data.modelType.value.lower() + '_model']
            prediction = model.predict(tv_spend)
            rmse = model.get('rmse', 'N/A')  # Retrieve RMSE if available

        # Create and return the prediction response
        response = PredictionResponse(
            tv_spend=float(data.tv_spend),
            predicted_sales=float(prediction[0]),
            modelType=data.modelType.value,
            modelRmse=str(rmse)
        )

        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)