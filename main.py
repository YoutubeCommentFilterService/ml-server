from fastapi import FastAPI, HTTPException
from typing import List, Tuple
from dotenv import load_dotenv
import pandas as pd
import redis.asyncio as redis

import os
import time
import hashlib

import torch
from helpers import TransformerClassificationModel, S3Helper
from schemes.fastapi_types import PredictItem,  PredictResult, PredictRequest, PredictResponse, PredictClassResponse
from schemes.config import (
    REDIS_REQUEST_TIME_KEY, 
    REDIS_LAST_REQUEST_TIME_KEY,
    REDIS_LAST_UPDATE_TIME_KEY,
    REDIS_MODEL_VERSION_KEY
)
from helpers.text_preprocessing import TextNormalizator

project_root_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(project_root_dir, "env", ".env"))
google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")
do_not_download_list = ['dataset-backup']
google_client_key_path = os.path.join(project_root_dir, 'env', 'ml-server-key.json')

redis_client = redis.Redis()
DEFAULT_TTL_EX = 30
DEFAULT_IDLE_TIME = 2

pid = str(os.getpid())

def get_file_version_hash():
    stat = os.stat('./model/dataset.csv')
    key = f"{stat.st_mtime}-{stat.st_size}".encode("utf-8")
    return hashlib.md5(key).hexdigest()

async def set_model_version():
    file_version = get_file_version_hash()
    await set_redis_key_value(REDIS_MODEL_VERSION_KEY, file_version)

async def get_model_version():
    return await redis_client.get(REDIS_MODEL_VERSION_KEY)

helper = S3Helper(project_root_dir, 'youtube-comment-predict')
nickname_model, comment_model, text_normalizator = None, None, None
model_type = None

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def waiting_for_idle(redis_key: str):
    redis_key = redis_key + pid
    while True:
        last_request_time = await redis_client.get(redis_key)
        if last_request_time is None:
            break
        if time.time() - last_request_time > DEFAULT_IDLE_TIME:
            break
        await asyncio.sleep(1)

async def update_last_update_time():
    while True:
        await set_redis_key_value(REDIS_LAST_UPDATE_TIME_KEY + pid, time.time(), ttl_ex=DEFAULT_TTL_EX)
        await asyncio.sleep(1)

async def update_last_request_time():
    while True:
        cur_time = time.time()
        await set_redis_key_value(REDIS_REQUEST_TIME_KEY, cur_time, ttl_ex=DEFAULT_TTL_EX)
        await set_redis_key_value(REDIS_LAST_REQUEST_TIME_KEY + pid, cur_time, ttl_ex=DEFAULT_TTL_EX)
        await asyncio.sleep(1)

async def set_redis_key_value(key, value, ttl_ex=None):
    if ttl_ex is not None:
        await redis_client.set(key, value, ex=ttl_ex)
    else:
        await redis_client.set(key, value)

async def check_model_updatable():
    global model_version
    while True:
        next_model_version = await get_model_version()
        if model_version != next_model_version:

            await waiting_for_idle(REDIS_LAST_REQUEST_TIME_KEY)

            task = asyncio.create_task(update_last_update_time())
            nickname_model.reload()
            comment_model.reload()
            text_normalizator.reload()
            read_predict_classes()
        
            task.cancel()

            print(f'({pid:>6}) update finish!', flush=True)
        model_version = next_model_version
        await asyncio.sleep(1 * 60)

def read_predict_classes():
    global nickname_predict_class, comment_predict_class
    classes = pd.read_csv('./model/dataset.csv', usecols=['nickname_class', 'comment_class'])
    nickname_predict_class = classes['nickname_class'].dropna().unique().tolist()
    comment_predict_class = classes['comment_class'].dropna().unique().tolist()

update_redis_task = None
async def startup():
    global update_redis_task, model_version, model_type, nickname_model, comment_model, text_normalizator

    if torch.cuda.is_available():
        fp = os.getenv('FP')
        fp = fp if fp is not None else 'fp32'
        comment_model = TransformerClassificationModel(model_type="comment", quantize=fp)
        nickname_model = TransformerClassificationModel(model_type="nickname", quantize=fp)

        do_not_download_list.extend(['comment_onnx', 'nickname_onnx'])

        print(f"({pid:>6}) Transformer loaded", flush=True)

        model_type = 'transformer'
    else:
        raise ValueError("CUDA not available")

    read_predict_classes()
    model_version = await get_model_version()

    nickname_model.load()
    comment_model.load()
    text_normalizator = TextNormalizator(normalize_file_path='./tokens/text_preprocessing.json', emoji_path='./tokens/emojis.txt', tokenizer_path='./model/tokenizer')

    update_redis_task = asyncio.create_task(check_model_updatable())

async def shutdown():
    nickname_model.unload()
    comment_model.unload()

    if update_redis_task is not None:
        update_redis_task.cancel()

    await redis_client.close()

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
    response = PredictClassResponse(
        nickname_predict_class=nickname_predict_class,
        comment_predict_class=comment_predict_class
    )
    return response

@app.post("/predict")
async def predict_batch(data: PredictRequest):
    # 업데이트중인 경우 대기 로직
    await waiting_for_idle(REDIS_LAST_UPDATE_TIME_KEY)
    print(f'({pid:>6}) predict request accepted... ', flush=True)

    redis_update_task = asyncio.create_task(update_last_request_time())
    
    try:
        items = data.items
        response_data = []

        if len(items) > 0:
            df = pd.DataFrame([{'nickname': item.nickname, 'comment': item.comment} for item in items])
            text_normalizator.run_text_preprocessing(df)

            nicknames = df['nickname'].tolist()
            comments = df['comment'].tolist()

            start = time.time()
            (nickname_outputs, nickname_categories) = nickname_model.predict(nicknames)
            (comment_outputs, comment_categories) = comment_model.predict(comments)

            # (nickname_outputs, nickname_categories), (comment_outputs, comment_categories) = await predict_process(nicknames, comments)
            print(f"({pid:>6}) predict len: {len(items)}, time: {time.time() - start}", flush=True)

            for comment_output, nickname_output in zip(comment_outputs, nickname_outputs):
                response_data.append(
                    PredictResult(
                        nickname_predicted=nickname_output[0],
                        nickname_predicted_prob=nickname_output[1],
                        comment_predicted=comment_output[0],
                        comment_predicted_prob=comment_output[1]
                    )
                )
        
        # 결과 반환
        return PredictResponse(items=response_data, model_type=model_type, nickname_categories=nickname_categories, comment_categories=comment_categories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        redis_update_task.cancel()
        try:
            await redis_update_task
        except Exception as e:
            print(e, flush=True)
        except asyncio.CancelledError as e:
            print(e, flush=True)

@app.patch("/update")
async def update_dataset():
    global model_version
    print(f'({pid:>6}) update start!', flush=True)
    try:
        helper.download()
        print(f'({pid:>6}) download model finished!', flush=True)
        await set_model_version()
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
