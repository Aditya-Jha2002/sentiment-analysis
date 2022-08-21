from pydantic import BaseModel

class PredRequest(BaseModel):
    text: str

class PredResponse(BaseModel):
    sentiment: bool
    class Config:
        orm_mode = True