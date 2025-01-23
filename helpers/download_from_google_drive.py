from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from dotenv import load_dotenv

import os
import json
import io

'''
https://github.com/hansheng0512/google-drive-and-python
이곳의 download.py 메서드를 사용
'''
class DownloadFromGoogleDrive:
    def __init__(self, root_dir):
        self.__ROOT_DIR = root_dir
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, 'model')

        load_dotenv(dotenv_path=os.path.join(self.__ROOT_DIR, 'env', '.env'))
        self.__DATASET_DRIVE_URL = os.getenv("DATASET_BASE_URL")

        with open(os.path.join(self.__ROOT_DIR, 'env/ml-server-key.json'), 'rb') as token:
            credential_info = json.load(token)

        credentials = service_account.Credentials.from_service_account_info(credential_info)
        self.__service = build("drive", "v3", credentials = credentials)

        self.__model_folder_ids = {
            "comment": os.getenv("COMMENT_MODEL_BASE_URL"),
            "nickname": os.getenv("NICKNAME_MODEL_BASE_URL")
        }

    def download_models(self):
        if not os.path.isdir(self.__MODEL_DIR):
            os.mkdir(self.__MODEL_DIR)
        
        self.download_dataset()

        for (_, folder_id) in self.__model_folder_ids.items():
            folder = self.__service.files().get(fileId=folder_id).execute()
            folder_name = folder.get("name")
            page_token = None
            
            results = (
                self.__service.files()
                    .list(
                        q=f"'{folder_id}' in parents",
                        spaces="drive",
                        fields="nextPageToken, files(id, name, mimeType)"
                    )
                    .execute()
            )
            page_token = results.get("nextPageToken", None)
            if page_token is None:
                items = results.get("files", [])
                for item in items:
                    bfolderpath = os.path.join(self.__MODEL_DIR, folder_name)
                    if not os.path.isdir(bfolderpath):
                        os.mkdir(bfolderpath)

                    filepath = os.path.join(bfolderpath, item["name"])
                    self.__download_files(item["id"], filepath)

        print("Download Model Finished!")

    def download_dataset(self):
        results = (
            self.__service.files()
                .list(
                    q=f"'{self.__DATASET_DRIVE_URL}' in parents",
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)"
                )
                .execute()
        )
        page_token = results.get("nextPageToken", None)
        if page_token is None:
            items = results.get("files", [])
            for item in items:
                if item["mimeType"] == 'text/csv':
                    if not os.path.isdir(self.__MODEL_DIR):
                        os.mkdir(self.__MODEL_DIR)
                    filepath = os.path.join(self.__MODEL_DIR, item["name"])
                    self.__download_files(item["id"], filepath)

    def __download_files(self, dowid, dfilespath):
        request = self.__service.files().get_media(fileId=dowid)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()

        with io.open(dfilespath, "wb") as f:
            fh.seek(0)
            f.write(fh.read())

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    downloader = DownloadFromGoogleDrive(root_dir)
    downloader.download_dataset()
    downloader.download_models()