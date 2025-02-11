from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from dotenv import load_dotenv
try:
    import torch
except:
    None

import os
import json
import io
from tqdm import tqdm

'''
https://github.com/hansheng0512/google-drive-and-python
이곳의 download.py 메서드를 사용

https://developers.google.com/drive/api/quickstart/python?hl=ko
공식 레퍼런스
'''

class DownloadFromGoogleDrive():
    def __init__(self, project_root_dir: str, model_folder_id: str, test_mode=False):
        with open(os.path.join(project_root_dir, 'env', 'ml-server-key.json'), 'rb') as token:
            credential_info = json.load(token)
        credentials = service_account.Credentials.from_service_account_info(credential_info)
        self._service = build('drive', 'v3', credentials=credentials)

        self._do_not_download = ['dataset-backup']
        if not test_mode:
            if torch.cuda.is_available():
                self._do_not_download.extend(['comment_onnx', 'nickname_onnx'])
            else:
                self._do_not_download.extend(['comment_model', 'nickname_model'])

        self._model_download_root_dir = os.path.join(project_root_dir, os.getenv("MODEL_SAVE_DIR_NAME"))

        self._model_folder_id = model_folder_id

        self.GOOGLE_DRIVE_FOLDER_TYPE = 'application/vnd.google-apps.folder'

    def _mkdir(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def _get_folders_data(self, folder_id: str):
        return self._service.files().list(
            q=f"'{folder_id}' in parents",
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
        ).execute()

    def _recursive(self, folder_id: str, path: str='root'):
        items = self._get_folders_data(folder_id)
        for item in items.get('files'):
            item_name = item.get('name')
            item_id = item.get('id')
            if item_name in self._do_not_download:
                continue
            curr_path = f'''{path}/{item_name}'''
            if item.get('mimeType') == self.GOOGLE_DRIVE_FOLDER_TYPE:
                self._mkdir(curr_path)
                self._recursive(item_id, curr_path)
            else:
                self._download_file(path, item)
    
    def download(self):
        print('download from google drive started...')
        if not os.path.exists(self._model_download_root_dir):
            os.mkdir(self._model_download_root_dir)
        self._recursive(self._model_folder_id, self._model_download_root_dir)
        print('download from google drive finished!')

    def _download_file(self, target_dir: str, file_info):
        file_id = file_info.get('id')
        file_name = file_info.get('name')

        request = self._service.files().get_media(fileId=file_id)

        downloaded_file = io.BytesIO()
        downloader = MediaIoBaseDownload(fd=downloaded_file, request=request)
        done = False

        desc = f'''{file_name:<30} '''
        pbar = tqdm(ncols=150, desc=desc)
        while done is False:
            status, done = downloader.next_chunk()
            if pbar.total is None:
                pbar.total = status.total_size
            
            if status:
                pbar.update(int(status.progress() * status.total_size) - pbar.n)
        pbar.close()

        save_path = f'''{target_dir}/{file_name}'''
        with open(save_path, 'wb') as localfile:
            localfile.write(downloaded_file.getvalue())

if __name__ == "__main__":
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(curr_dir, '..'))
        load_dotenv(os.path.join(root_dir, 'env', '.env'))
        downlader = DownloadFromGoogleDrive(project_root_dir=root_dir,
                                            model_folder_id=os.getenv('MODEL_ROOT_FOLDER_ID'),
                                            test_mode=True)
        downlader.download()
    except Exception as e:
        print(e.message)
