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

fp = os.environ.get('FP')

def check_all_file_exists(check_file_or_dirs: list[str]) -> dict[str, bool]:
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
    tokenizer = AutoTokenizer.from_pretrained('model/tokenizer')
    model_subfix = '_fp16' if fp == 'fp16' else ''
    nickname = AutoConfig.from_pretrained('model/nickname_model' + model_subfix)
    comment = AutoConfig.from_pretrained('model/comment_model' + model_subfix)

    return {
        'nickname_model' + model_subfix: len(tokenizer) == nickname.vocab_size,
        'comment_model' + model_subfix : len(tokenizer) == comment.vocab_size
    }

def when_ready(server):
    project_root_dir = os.path.dirname(os.path.abspath(__file__))

    load_dotenv(os.path.join(project_root_dir, "env", ".env"))

    print('Download Neccesarry Files Start...', flush=True)
    helper = S3Helper(project_root_dir, 'youtube-comment-predict')
    if not is_exist_model():
        helper.download()
    else:
        helper.download(['dataset.csv'])
    print('Download Neccesarry Files Finished!', flush=True)

    check_file_or_dirs = ['dataset.csv', 'tokenizer']
    check_file_or_dirs.extend(
        model + ('_fp16' if fp == 'fp16' else '') 
        for model in ['nickname_model', 'comment_model']
    )

    missing_files = [ k for k, v in check_all_file_exists(check_file_or_dirs).items() if not v ]
    if missing_files:
        raise RuntimeError(f'필요한 모델/파일이 없습니다: {missing_files}')
    
    incorrect_vocab = [ k for k, v in is_vocab_correct().items() if not v ]
    if incorrect_vocab:
        raise RuntimeError(f'해당 model의 vocab_size가 올바르지 않습니다: {incorrect_vocab}')

    print('Loading Redis Client Start...', flush=True)
    client = Redis()
    file_hash = get_model_version_hash()
    client.set(REDIS_MODEL_VERSION_KEY, file_hash)
    print('Loading Redis Client Finished!', flush=True)

    print('master ready!')

def is_exist_model():
    if not os.path.exists('./model'):
        return False
    files = [re.sub(r'_[a-z0-9]+', '', f) for f in os.listdir('./model') if f != 'dataset.csv']
    return len(set(files)) == 3

def get_model_version_hash():
    stat = os.stat('./model/dataset.csv')
    key = f"{stat.st_mtime}-{stat.st_size}".encode("utf-8")
    return hashlib.md5(key).hexdigest()