from fastapi import FastAPI
from prediction_service import predict, schemas
app = FastAPI()

@app.post("/")
def post_preds(request_dict: schemas.PredRequest, response_model=schemas.PredResponse):
    sentiment = predict.api_response(request_dict)
    response = {"sentiment": sentiment}
    return response