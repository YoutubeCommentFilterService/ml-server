from helpers import S3Helper
import os
from dotenv import load_dotenv
import hashlib
from redis import Redis
from schemes.config import REDIS_MODEL_VERSION_KEY
import re

bind="0.0.0.0:5000"
workers=3
worker_class="uvicorn.workers.UvicornWorker"
timeout=120

def when_ready(server):
    project_root_dir = os.path.dirname(os.path.abspath(__file__))

    load_dotenv(os.path.join(project_root_dir, "env", ".env"))
    helper = S3Helper(project_root_dir, 'youtube-comment-predict')
    if not is_exist_model():
        helper.download()
    else:
        helper.download(['dataset.csv'])

    client = Redis()
    file_hash = get_model_version_hash()
    client.set(REDIS_MODEL_VERSION_KEY, file_hash)

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