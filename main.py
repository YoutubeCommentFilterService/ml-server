from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

import torch

import re
import os

from helpers import ONNXClassificationModel, TransformerClassificationModel, GoogleDriveHelper

project_root_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(project_root_dir, "env", ".env"))
google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")
do_not_download_list = ['dataset-backup']
google_client_key_path = os.path.join(project_root_dir, 'env', 'ml-server-key.json')

if torch.cuda.is_available():
    comment_model = TransformerClassificationModel(model_type="comment")
    nickname_model = TransformerClassificationModel(model_type="nickname")
else:
    comment_model = ONNXClassificationModel(model_type="comment")
    nickname_model = ONNXClassificationModel(model_type="nickname")

helper = GoogleDriveHelper(project_root_dir=project_root_dir,
                           google_client_key_path=google_client_key_path,
                           google_drive_owner_email=google_drive_owner_email,
                           do_not_download_list=do_not_download_list,
                           local_target_root_dir_name='model',
                           drive_root_folder_name='comment-filtering')

if not os.path.exists('./model'):
    helper.download_all_files()

app = FastAPI()

class PredictItem(BaseModel):
    id: str
    nickname: str
    comment: str
    
class PredictResult(BaseModel):
    id: str
    nickname_original: str
    nickname_predicted: str
    comment_original: str
    comment_predicted: str

class PredictRequest(BaseModel):
    items: List[PredictItem]

class PredictResponse(BaseModel):
    items: List[PredictResult]

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
        items = data.items
        response_data = []

        for item in items:
            comment = item.comment
            comment = re.sub(r'\d+:(?:\d+:?)?\d+', '[TIME]', comment)

            nickname = item.nickname
            nickname = re.sub(r'[-._]', ' ', nickname)

            comment_output = comment_model.predict(comment)
            nickname_output = nickname_model.predict(nickname)

            response_data.append(PredictResult(id=item.id, 
                                               nickname_original=item.nickname,
                                               nickname_predicted=nickname_output,
                                               comment_original=item.comment,
                                               comment_predicted=comment_output))
        
        # 결과 반환
        return PredictResponse(items=response_data)
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
    
@app.patch("/update")
def update_dataset():
    try:
        downloader.download()
        nickname_model.reload()
        comment_model.reload()
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
