from fastapi import FastAPI, HTTPException
import pandas as pd

from src.api.schema import ChurnRequest, ChurnResponse
from src.pipeline.inference_pipeline import InferencePipeline

app = FastAPI(
    title = "Customer Churn Prediction API",
    description = "Production-grade ML API with threshold based API inference.",
    version = "1.0.0"
)

#Load inference pipeline
inference_pipeline = InferencePipeline()

@app.get("/")
def health_check():
    return {
        "status" : "ok",
        "message": "Churn Prediction API is running."
    }

@app.post("/predict", response_model =ChurnResponse)
def predict_churn(request : ChurnRequest):
    try:
        #Convert request to Dataframe
        input_df= pd.DataFrame([request.dict()])

        #Run inference
        result = inference_pipeline.predict(input_df)

        return ChurnResponse(
            churn_probability=float(result["churn_probability"].iloc[0]),
            churn_prediction=str(result["churn_prediction"].iloc[0])
        )

    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))