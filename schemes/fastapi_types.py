from pydantic import BaseModel
from typing import List, Optional

class PredictItem(BaseModel):
    nickname: str
    comment: str
    
class PredictResult(BaseModel):
    nickname_predicted: str
    nickname_predicted_prob: List[float]
    comment_predicted: str
    comment_predicted_prob: List[float]

class PredictRequest(BaseModel):
    items: List[PredictItem]

class PredictResponse(BaseModel):
    items: List[PredictResult]
    model_type: Optional[str]
    nickname_categories: List[str]
    comment_categories: List[str]

class PredictClassResponse(BaseModel):
    nickname_predict_class: List[str]
    comment_predict_class: List[str]