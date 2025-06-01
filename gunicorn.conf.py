from helpers import S3Helper
import os
from dotenv import load_dotenv
import hashlib
from redis import Redis
from schemes.config import REDIS_MODEL_VERSION_KEY
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig

bind="0.0.0.0:5000"
workers=3
worker_class="uvicorn.workers.UvicornWorker"
timeout=120

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fp = os.environ.get('FP', '')
project_root_dir = os.path.dirname(os.path.abspath(__file__))

check_file_or_dirs = ['dataset.csv', 'tokenizer']
check_file_or_dirs.extend(
    model + ('_fp16' if fp == 'fp16' else '') 
    for model in ['nickname_model', 'comment_model']
)

def check_all_file_exists() -> dict[str, bool]:
    model_base_path = 'model/'
    return dict(
        map(
            lambda x: (x, is_dir_nonempty(model_base_path + x)), 
            check_file_or_dirs
        )
    )

def is_dir_nonempty(path: str) -> bool:
    p = Path(path)
    return any(p.iterdir()) if p.is_dir() else p.exists()

def is_vocab_correct() -> dict[str, bool]:
    model_subfix = '_fp16' if fp == 'fp16' else ''
    nickname_vocal_len = AutoConfig.from_pretrained('model/nickname_model' + model_subfix).vocab_size
    comment_vocal_len = AutoConfig.from_pretrained('model/comment_model' + model_subfix).vocab_size
    tokenizer_vocal_len = len(AutoTokenizer.from_pretrained('model/tokenizer'))

    return {
        'nickname_model' + model_subfix: tokenizer_vocal_len == nickname_vocal_len,
        'comment_model' + model_subfix : tokenizer_vocal_len == comment_vocal_len
    }

def download_neccessary_files():
    print('Download Neccesarry Files Start...', flush=True)
    helper = S3Helper(project_root_dir, 'youtube-comment-predict')
    try:
        check_all_neccessary_file_exists()
        helper.download(['dataset.csv'])
    except:
        # 모든 파일 다운로드
        helper.download()
    print('Download Neccesarry Files Finished!', flush=True)

def check_all_neccessary_file_exists():
    missing_files = [ k for k, v in check_all_file_exists().items() if not v ]
    if missing_files:
        raise RuntimeError(f'필요한 모델/파일이 없습니다: {missing_files}')
    
def check_vocab_is_correct():
    incorrect_vocab = [ k for k, v in is_vocab_correct().items() if not v ]
    if incorrect_vocab:
        raise RuntimeError(f'해당 model의 vocab_size가 올바르지 않습니다: {incorrect_vocab}')
    
def start_redis():
    print('Loading Redis Client Start...', flush=True)
    client = Redis()
    try:
        stat = os.stat('./model/dataset.csv')
        key = f"{stat.st_mtime}-{stat.st_size}".encode("utf-8")
        file_hash = hashlib.md5(key).hexdigest()
        client.set(REDIS_MODEL_VERSION_KEY, file_hash)
    finally:
        client.close()
    print('Loading Redis Client Finished!', flush=True)

def on_starting(server):
    load_dotenv(os.path.join(project_root_dir, "env", ".env"))

    download_neccessary_files()

    check_vocab_is_correct()
    start_redis()

    print("server started!")
    pass