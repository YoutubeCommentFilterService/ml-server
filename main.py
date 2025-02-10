from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

import re
import os

do_not_download_list = ['dataset-backup']

try:
    raise ValueError("For Testing!!")

    import torch
    from helpers import TransformerClassificationModel

    if torch.cuda.is_available():
        comment_model = TransformerClassificationModel(model_type="comment")
        nickname_model = TransformerClassificationModel(model_type="nickname")

        do_not_download_list.extend(['comment_onnx', 'nickname_onnx'])

        print("Transformer loaded")

        model_type = 'transformer'
    else:
        raise ValueError("CUDA not available")
except Exception as e:
    print(f"Error message: {str(e)}")

    try:
        from helpers import ONNXClassificationModel

        comment_model = ONNXClassificationModel(model_type="comment")
        nickname_model = ONNXClassificationModel(model_type="nickname")

        do_not_download_list.extend(['comment_model', 'nickname_model'])

        print("ONNX loaded")

        model_type = 'onnxruntime-gpu'
    except ImportError:
        raise ImportError("Neither TransformerClassificationModel nor ONNXClassificationModel could be imported")

from helpers import GoogleDriveHelper

project_root_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(project_root_dir, "env", ".env"))
google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")
do_not_download_list = ['dataset-backup']
google_client_key_path = os.path.join(project_root_dir, 'env', 'ml-server-key.json')

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
    nickname_predicted: str
    comment_predicted: str

class PredictRequest(BaseModel):
    items: List[PredictItem]

class PredictResponse(BaseModel):
    items: List[PredictResult]
    model_type: Optional[str]

async def startup():
    nickname_model.load()
    comment_model.load()

async def shutdown():
    nickname_model.unload()
    comment_model.unload()

app.add_event_handler("startup", startup)
app.add_event_handler("shutdown", shutdown)
    
@app.post("/predict")
def predict_batch(data: PredictRequest):
    try:
        items = data.items
        response_data = []

        nicknames = [item.nickname.replace('@', '') for item in items]
        nicknames = [re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9-_.]+", '[DEFAULT_NICK]', nickname) for nickname in nicknames]
        nicknames = [re.sub(r"-[a-zA-Z0-9]{3}$|-[a-zA-Z0-9]{5}$", '', nickname) for nickname in nicknames]
        nicknames = [re.sub(r'[-._]', ' ', nickname) for nickname in nicknames]
        nicknames = [nickname.strip() for nickname in nicknames]

        comments = [re.sub(r'https?:\/\/[^\s]+|www.[^\s]+', '[URL]', item.comment) for item in items]
        comments = [re.sub(r'#(\w+)', '[HASH_TAG]', comment) for comment in comments]
        comments = [re.sub(r'[’‘]+', "'", comment) for comment in comments]
        comments = [re.sub(r'[”“]+', '"', comment) for comment in comments]
        comments = [re.sub(r'[\*\^]', "", comment) for comment in comments]
        comments = [re.sub(r'\d+:(?:\d+:?)?\d+', '[TIME]', comment) for comment in comments]
        comments = [re.sub(r'chill', '칠', comment, flags=re.IGNORECASE) for comment in comments]
        comments = [comment.strip() for comment in comments]
        comments = [comment or "[EMPTY]" for comment in comments]

        nickname_outputs = nickname_model.predict(nicknames)
        comment_outputs = comment_model.predict(comments)

        for item, comment_output, nickname_output in zip(items, comment_outputs, nickname_outputs):
            response_data.append(PredictResult(id=item.id, 
                                               nickname_predicted=nickname_output,
                                               comment_predicted=comment_output))
        
        # 결과 반환
        return PredictResponse(items=response_data, model_type=model_type)
    
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
    
@app.patch("/update")
def update_dataset():
    try:
        helper.download_all_files()
        nickname_model.reload()
        comment_model.reload()
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
