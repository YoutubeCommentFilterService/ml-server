# DownloadFromGoogleDrive

## Introduction

파이썬에서 Google Drive의 파일을 다운받기 위한 방법을 설명해드립니다.

1. GCP에서 필요한 key file 생성 및 API 구독
2. 파이썬에서의 적용

으로 구성되어있습니다.

[공식 레퍼런스](https://developers.google.com/drive/api/quickstart/python?hl=ko)와 [타인의 오픈소스](https://github.com/hansheng0512/google-drive-and-python)를 참고 바랍니다.

> 주의!

현재 제 머신러닝 프로젝트에 맞게 설정되어있습니다. key의 경로, 다운로드 하지 않을 폴더 등은 적용하려는 프로젝트에 맞게 설정해주시면 되겠습니다.  
또한 하나의 디렉터리만 타겟으로 다운받도록 하였습니다.

추후 리팩토링을 통해 범용적으로 사용할 수 있도록 할 예정입니다!

## Settings

### GCP 프로젝트 설정 및 키 생성

1. 먼저 gcp 프로젝트를 만들어줍니다.

   ![GCP 콘솔에서 새 프로젝트 생성](img/01-gcp-create.png)

   ![새 프로젝트 생성](img/02-gcp-new-project.png)

2. 생성된 프로젝트에서 API 및 서비스를 들어갑니다.

   ![API 및 서비스로 이동](img/03-gcp-dashboard.png)

3. 서비스 계정을 생성합니다.

   ![사용자 인증 정보 - 서비스 계정 관리](img/04-gcp-service-account.png)

   ![서비스 계정 - 서비스 계정 만들기](img/05-gcp-create-service-account.png)

   ![서비스 계정 - 생성 페이지](img/06-gcp-create-service-account-page.png)

4. 이후 새로운 키를 생성합니다.

   > 생성한 서비스 계정을 누른 후 키로 이동합니다.

   ![새 키 만들기](img/07-create-new-key.png)

   ![키 타입](img/08-key-type.png)

   ![키 다운로드](img/09-key-downloaded.png)

   <details>
       <summary>json 키 형식</summary>

   ```json
   {
     "type": "service_account",
     "project_id": "youtube-comment-435905",
     "private_key_id": "9428e75******************06e2f3",
     "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIB******************************7yRMyL/g=\n-----END PRIVATE KEY-----\n",
     "client_email": "google-drive-api@youtube-comment-435905.iam.gserviceaccount.com",
     "client_id": "108************21910",
     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
     "token_uri": "https://oauth2.googleapis.com/token",
     "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
     "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/google-drive-api%40youtube-comment-435905.iam.gserviceaccount.com",
     "universe_domain": "googleapis.com"
   }
   ```

   </details>

5. 다시 콘솔로 돌아와 API로 이동합니다.

   ![사용 설정된 API 및 서비스](img/10-api-page.png)

6. Google Drive API를 검색 후 사용버튼을 눌러 활성화 합니다.

   ![Google Drive API 활성화](img/11-api-install.png)

### 파이썬에서 적용

7.  필요한 패키지를 받습니다.

    oauth 로그인이 아닌, 사용자 인증을 구글로부터 이미 받았기에 google-api-python-client만 필요합니다.

    ```
    pip install google-api-python-client

    pip install python-dotenv, torch, tqdm
        dotenv: 서버에서 실행할 환경을 env로 불러옴
        torch: 현재 서버가 cuda를 사용할 수 있는지 확인
        tqdm: 진행상황을 확인할 수 있는 모듈
    ```

8.  받은 secret key를 업로드하고 .env를 작성합니다.

    ```
    // .env

    MODEL_ROOT_FOLDER_ID=(root 폴더 id)
    MODEL_SAVE_DIR_NAME=(다운로드한 데이터를 저장할 root directory path)
    ```

    <details>
        <summary>MODEL_ROOT_FOLDER_ID 얻는 방법</summary>
        <img src="img/99-drive-root-folder.png" alt="구글 드라이브에서 root directory id를 찾는 방법" />

        folders path 뒤에 있는 문자열이 folder ID입니다.

    </details>

9.  service를 작성합니다.

    ```python
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    import json

    with open(os.path.join(project_root_dir, 'env', 'ml-server-key.json'), 'rb') as token:
        credential_info = json.load(token)
    credentials = service_account.Credentials.from_service_account_info(credential_info)
    service = build('drive', 'v3', credentials=credentials)
    ```

    이후 공식 문서에 맞게 사용하면 됩니다.

    <details>
        <summary>예제</summary>

    - 폴더에 있는 모든 object를 조회

      ```python
      service.files().list(
              q=f"'{folder_id}' in parents",
              spaces='drive',
              fields='nextPageToken, files(id, name, mimeType)',
          ).execute()
      ```

    - 파일 다운로드

      ```python
      request = service.files().get_media(fileId=file_id)

      downloaded_file = io.BytesIO()
      downloader = MediaIoBaseDownload(fd=downloaded_file, request=request)
      done = False

      while done is False:
          status, done = downloader.next_chunk()
      ```

    </details>

## Usage

```python
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, '..'))
load_dotenv(os.path.join(root_dir, 'env', '.env'))
downlader = DownloadFromGoogleDrive(project_root_dir=root_dir,
                                    model_folder_id=os.getenv('MODEL_ROOT_FOLDER_ID'),
                                    test_mode=True)
downlader.download()
```
