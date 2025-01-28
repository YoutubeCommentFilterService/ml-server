from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import torch

import re
import os

from helpers import ONNXClassificationModel, TransformerClassificationModel, DownloadFromGoogleDrive

project_root_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(project_root_dir, "env", ".env"))

if torch.cuda.is_available():
    comment_model = TransformerClassificationModel(model_type="comment")
    nickname_model = TransformerClassificationModel(model_type="nickname")
else:
    comment_model = ONNXClassificationModel(model_type="comment")
    nickname_model = ONNXClassificationModel(model_type="nickname")

downloader = DownloadFromGoogleDrive(project_root_dir=project_root_dir,
                                     model_folder_id=os.getenv('MODEL_ROOT_FOLDER_ID'))

app = FastAPI()

class PredictRequest(BaseModel):
    nickname: str
    comment: str

async def startup():
    nickname_model.load()
    comment_model.load()

async def shutdown():
    nickname_model.unload()
    comment_model.unload()

app.add_event_handler("startup", startup)
app.add_event_handler("shutdown", shutdown)


@app.post("/predict")
def predict(data: PredictRequest):
    try:
        # 요청에서 텍스트 데이터 가져오기
        comment = data.comment
        nickname = data.nickname
        nickname = re.sub(r'[-._]', ' ', nickname)
        
        # 예측 수행
        comment_output = comment_model.predict(comment)
        nickname_output = nickname_model.predict(nickname)
        
        # 결과 반환
        return {
            'status': 'success',
            'nickname': nickname_output,
            'comment': comment_output
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
    
@app.patch("/update")
def update_dataset():
    try:
        None
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
