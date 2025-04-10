from fastapi import FastAPI, HTTPException
from typing import List, Tuple
from dotenv import load_dotenv
import pandas as pd
import redis
    
import time

import re
import os
import traceback
import unicodedata

import torch
from helpers import TransformerClassificationModel, S3Helper
from schemes.fastapi_types import PredictItem,  PredictResult, PredictRequest, PredictResponse, PredictClassResponse
from schemes.config import REDIS_PUBSUB_TEGRA_KEY, REDIS_PUBSUB_TEGRA_MAX_VALUE, REDIS_PUBSUB_UPDATE_KEY, REDIS_REQUEST_TIME_KEY
from helpers.text_preprocessing import run_text_preprocessing, replace_regex_predict_data

do_not_download_list = ['dataset-backup']

project_root_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(project_root_dir, "env", ".env"))
google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")
do_not_download_list = ['dataset-backup']
google_client_key_path = os.path.join(project_root_dir, 'env', 'ml-server-key.json')

redis_client = redis.Redis()

helper = S3Helper(project_root_dir, 'youtube-comment-predict')
if not os.path.exists('./model'):
    helper.download()

if torch.cuda.is_available():
    fp = os.getenv('FP')
    fp = fp if fp is not None else 'fp32'
    comment_model = TransformerClassificationModel(model_type="comment", quantize=fp)
    nickname_model = TransformerClassificationModel(model_type="nickname", quantize=fp)

    do_not_download_list.extend(['comment_onnx', 'nickname_onnx'])

    print("Transformer loaded")

    model_type = 'transformer'
else:
    raise ValueError("CUDA not available")

app = FastAPI()

# origins = [
#     "https://api.spampredict.store",
#     "https://spampredict.store"
# ]
# app.add_middleware(CORSMiddleware,
#                    allow_origins=origins,
#                    allow_credentials=True,
#                    allow_methods=["*"],
#                    allow_headers=["*"])

nickname_predict_class = None
comment_predict_class = None
power_mode = {"max": None, "min": None}
is_idle = True

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

is_updating = False
def subscribe_redis():
    global is_updating
    pubsub = redis_client.pubsub()
    pubsub.subscribe(REDIS_PUBSUB_UPDATE_KEY)

    for message in pubsub.listen():
        if message['channel'].decode('utf-8') == REDIS_PUBSUB_UPDATE_KEY and message['type'] == 'message':
            while not is_idle:
                time.sleep(1)

            is_updating = True
            nickname_model.reload()
            comment_model.reload()
            is_updating = False

            print('update finish!')

import threading

async def startup():
    global nickname_predict_class, comment_predict_class, power_mode
    classes = pd.read_csv('./model/dataset.csv', usecols=['nickname_class', 'comment_class'])
    nickname_predict_class = classes['nickname_class'].dropna().unique().tolist()
    comment_predict_class = classes['comment_class'].dropna().unique().tolist()

    del classes

    nickname_model.load()
    comment_model.load()

    threading.Thread(target=subscribe_redis, daemon=True).start()

async def shutdown():
    nickname_model.unload()
    comment_model.unload()

app.add_event_handler("startup", startup)
app.add_event_handler("shutdown", shutdown)

import asyncio

async def predict_process(nicknames: List[str], comments: List[str]) -> Tuple[PredictResult, PredictResult]:
    nickname_result, comment_result = await asyncio.gather(
        asyncio.to_thread(nickname_model.predict, nicknames),
        asyncio.to_thread(comment_model.predict, comments),
    )
    return nickname_result, comment_result

@app.get("/predict-category")
async def get_predict_category():
    response = PredictClassResponse(nickname_predict_class=nickname_predict_class, 
                                    comment_predict_class=comment_predict_class)
    return response

@app.post("/predict")
async def predict_batch(data: PredictRequest):
    global is_idle, is_updating

    async def keep_updating_redis():
        while not is_idle:
            redis_client.set(REDIS_REQUEST_TIME_KEY, time.time())
            await asyncio.sleep(1)

    while is_updating:
        await asyncio.sleep(1)
    print('predict request accepted...')
    is_idle = False

    redis_client.publish(REDIS_PUBSUB_TEGRA_KEY, REDIS_PUBSUB_TEGRA_MAX_VALUE)
    redis_update_task = asyncio.create_task(keep_updating_redis())
    
    try:
        items = data.items
        response_data = []

        nickname_categories, comment_categories = nickname_predict_class, comment_predict_class

        if len(items) > 0:
            df = pd.DataFrame([{'nickname': item.nickname, 'comment': item.comment} for item in items])
            run_text_preprocessing(df, './tokens/emojis.txt')
            # replace_regex_predict_data(df)

            nicknames = df['nickname'].tolist()
            comments = df['comment'].tolist()

            start = time.time()
            (nickname_outputs, nickname_categories), (comment_outputs, comment_categories) = await predict_process(nicknames, comments)
            print(f"predict len: {len(items)}, time: {time.time() - start}")

            print(len(items), len(comment_outputs), len(nickname_outputs))

            index = 0
            for item, comment_output, nickname_output in zip(items, comment_outputs, nickname_outputs):
                if nickname_output is None:
                    print(f'\tnickname{index} = {nickname_output}, {item}')
                if comment_output is None:
                    print(f'\tcomment{index} = {comment_output}, {item}')
                index = index + 1
                
                response_data.append(PredictResult(nickname_predicted=nickname_output[0],
                                                   nickname_predicted_prob=nickname_output[1],
                                                   comment_predicted=comment_output[0],
                                                   comment_predicted_prob=comment_output[1]))
        
        # 결과 반환
        return PredictResponse(items=response_data, model_type=model_type, nickname_categories=nickname_categories, comment_categories=comment_categories)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        is_idle = True
        redis_update_task.cancel()
    
@app.patch("/update")
def update_dataset():
    print('update start!')
    try:
        helper.download()
        redis_client.publish(REDIS_PUBSUB_UPDATE_KEY, '')
        return 'update model succeed'
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@app.get("/status")
def get_server_status():
    status = {
        "model": model_type,
        "comment": True if comment_model is not None else False,
        "nickname": True if comment_model is not None else False,
        "downloader": True if helper is not None else False,
    }
    return status

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
