from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from helpers import TextClassificationModel, DownloadFromGoogleDrive

import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "env/.env"))

app = FastAPI()

comment_model = TextClassificationModel(BASE_DIR, "comment")
nickname_model = TextClassificationModel(BASE_DIR, "nickname")
downloader = DownloadFromGoogleDrive(BASE_DIR)

@app.get("/")
def root():
    return {"hello": "world"}

class EvalData(BaseModel):
    id: str
    nickname: str
    comment: str

class EvalDataResponse(BaseModel):
    id: str
    result: list

def discard_at_char(nickname: str) -> str:
    return nickname[1:].replace("-", "_")

def nomalize_comment(comment: str) -> str:
    comment = re.sub(r'[\r\n,]+', ' ', comment)
    comment = re.sub(r'[^가-힣a-zA-Z0-9~!@#$%^&*()_+\-=\[\]{}:;"\'<>,.?/\s]', '', comment)
    comment = re.sub(r'(\.\.\.)\.+', '...', comment)
    comment = re.sub(r'(ㅋㅋㅋ)ㅋ+', 'ㅋㅋㅋ', comment)
    comment = re.sub(r'(ㅠㅠㅠ)ㅠ+', 'ㅠㅠㅠ', comment)
    comment = re.sub(r'(ㅜㅜㅜ)ㅜ+', 'ㅜㅜㅜ', comment)
    comment = re.sub(r'(ㄱㄱㄱ)ㄱ+', 'ㄱㄱㄱ', comment)
    comment = re.sub(r'(\?\?\?)\?+', '???', comment)
    comment = re.sub(r'\b(\d{1,3}):(\d{1,2}):(\d{1,2}):(\d{1,2})\b|\b(\d{1,3}):(\d{1,2}):(\d{1,2})\b|\b(\d{1,3}):(\d{1,2})\b', '', comment)

    return comment

@app.post("/eval")
def eval_comment_and_nickname(body: EvalData):
    comment: str = nomalize_comment(body.comment)
    nickname: str = discard_at_char(body.nickname)
    result = []
    result.append(comment_model.eval(comment))
    result.append(nickname_model.eval(nickname))
    # result = list(filter(lambda x: x != "정상", result))

    return EvalDataResponse(id=body.id, result=result)

@app.patch("/")
def update_model():
    downloader.download_models()
    comment_model.reload()
    nickname_model.reload()