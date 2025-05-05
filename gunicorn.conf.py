from helpers import S3Helper
import os
from dotenv import load_dotenv
import hashlib
from redis import Redis
from schemes.config import REDIS_MODEL_VERSION_KEY

bind="0.0.0.0:5000"
workers=5
worker_class="uvicorn.workers.UvicornWorker"

def when_ready(server):
    project_root_dir = os.path.dirname(os.path.abspath(__file__))

    load_dotenv(os.path.join(project_root_dir, "env", ".env"))
    helper = S3Helper(project_root_dir, 'youtube-comment-predict')
    if not os.path.exists('./model') or not any(os.listdir('./model')):
        helper.download()

    client = Redis()
    file_hash = get_model_version_hash()
    client.set(REDIS_MODEL_VERSION_KEY, file_hash)

    print('master ready!')

def get_model_version_hash():
    stat = os.stat('./model/dataset.csv')
    key = f"{stat.st_mtime}-{stat.st_size}".encode("utf-8")
    return hashlib.md5(key).hexdigest()