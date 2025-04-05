# Introduction

## 디렉터리 구조

```
.
├── README.md
├── config.json
├── env
│   ├── .env                                    # 실행에 필요한 환경변수를 담아뒀습니다.
│   └── ml-server-key.json                      # google drive에 접근하기 위한 서비스 계정입니다.
├── helpers
│   ├── __init__.py                             # main.py에서 불러오기 편하도록 작성한 init입니다.
│   ├── download_from_google_drive.py
│   ├── onnx_classification_model.py
│   └── transformer_classification_model.py
├── main.py
└── model                                       # 학습한 모델들이 담기는 디렉터리입니다.
```

## 패키지 설치

- x86_64 환경

  > pip install -r requirements_x86-64.txt

- arm64 환경

  > pip install -r requirements_arm64.txt

- 직접 설치
  > 서버: fastapi, uvicorn[standard]
  > 모델 다운로드: boto3, google-api-python-client, tqdm
  > 모델 로더: transformers
  > 메시지큐: redis

## tegra, 즉 jetson 시리즈에서 구동

아래의 내용을 추가해야 sudo를 입력하지 않아도 설정을 바꿀 수 있습니다!

```bash
// /etc/sudoers
{{ username }} ALL=NOPASSWD: /usr/sbin/nvpmodel
```

# Helper

## 텍스트 클래스 분류기

### Introduction

> 두 모델은 동일한 메서드를 지원합니다!

load로 model과 tokenizer를 불러오고, predict로 추론하면 됩니다.

실행 환경에 따라 선택적으로 사용하게 됩니다.

- TransformerClassificationModel

  > CUDA를 사용할 수 있는 경우. 성능이 좋습니다.

- ONNXClassificationModel
  > CUDA를 사용할 수 없는 경우. onnx를 양자화하여 실행 시간을 줄였습니다.

### Usage

```python
root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nickname_model=TransformerClassificationModel(model_type="nickname",
                                                base_path=root_dir+'/model')
nickname_model=ONNXClassificationModel(model_type="nickname",
                                                base_path=root_dir+'/model')
nickname_model.load()

text=input("검증할 닉네임을 입력해주세요: ")
nickname=re.sub(r"-._", " ", text)
predicted=nickname_model.predict(text)
print(predicted, type(nickname_model.predict(text)))
```

## DownloadFromGoogleDrive

> 이 클래스는 Deprecated 되었습니다. GoogleDriveHelper를 사용해주시기 바랍니다.

### Introduction

파이썬에서 Google Drive의 파일을 다운받는 helper class입니다.

1. GCP에서 필요한 key file 생성 및 API 구독
2. 파이썬에서의 적용

으로 구성되어있습니다.

[공식 레퍼런스](https://developers.google.com/drive/api/quickstart/python?hl=ko)와 [타인의 오픈소스](https://github.com/hansheng0512/google-drive-and-python)를 참고 바랍니다.

> 주의!

현재 제 머신러닝 프로젝트에 맞게 설정되어있습니다. key의 경로, 다운로드 하지 않을 폴더 등은 적용하려는 프로젝트에 맞게 설정해주시면 되겠습니다.  
또한 하나의 디렉터리만 타겟으로 다운받도록 하였습니다.

추후 리팩토링을 통해 범용적으로 사용할 수 있도록 할 예정입니다!

### Usage

```python
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, '..'))
load_dotenv(os.path.join(root_dir, 'env', '.env'))
downlader = DownloadFromGoogleDrive(project_root_dir=root_dir,
                                    model_folder_id=os.getenv('MODEL_ROOT_FOLDER_ID'),
                                    test_mode=True)
downlader.download()
```

## GoogleDriveHelper

### Intdoduction

위에서 작성한 `DownloadFromGoogleDrive`를 리팩토링한 클래스입니다.

upload, download, delete를 지원합니다.

파일 및 폴더 업로드 후 발생한 파일 및 폴더가 나타나지 않음을 해결하였습니다.

```
업로드 후 해당 id를 가진 파일의 권한을 writer로 합니다.
google service account는 하나의 계정으로 취급되어, drive owner과는 다른 drive를 사용합니다.
```

### Settings

#### GCP 프로젝트 설정 및 키 생성

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

#### 구글 드라이브의 폴더 권한 공유

7. Google Drive에서 폴더의 접근 권한을 설정합니다.

   ![Google Drive의 폴더에서 google service account의 액세스 권한을 공유](img/12-google-drive-share-option.png)

#### 파이썬에서 적용

8.  필요한 패키지를 받습니다.

    oauth 로그인이 아닌, 사용자 인증을 구글로부터 이미 받았기에 google-api-python-client만 필요합니다.

    ```
    pip install --upgrade google-api-python-client

    pip install python-dotenv, torch, tqdm
        dotenv: 서버에서 실행할 환경을 env로 불러옴
        torch: 현재 서버가 cuda를 사용할 수 있는지 확인
        tqdm: 진행상황을 확인할 수 있는 모듈
    ```

9.  받은 secret key를 업로드하고 .env를 작성합니다.

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

10. service를 작성합니다.

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

11. 업로드한 파일의 권한을 수정하는 코드를 작성합니다.

    owner를 설정하는 것은 권한 오류로 인해 불가능합니다. 실제로 권한이 있는 key file을 썼음에도 403 error가 발생하였습니다.  
    따라서 writer로 바꾸는 편법을 사용하여 드라이브를 가진 유저가 삭제할 수 있도록 합니다.

    외부의 다른 사용자가 보지 못하게 하고싶다면, permission_reader를 설정하는 부분을 주석처리하면 됩니다.

    ```python
    def _change_permission(self, file_id:str):
        # permission_owner = {'type': 'user', 'role': 'owner', 'emailAddress': self.google_drive_owner_email}
        permission_writer = {'type': 'user', 'role': 'writer', 'emailAddress': self.google_drive_owner_email}
        permission_reader = {'type': 'anyone', 'role': 'reader'}

        self._permission_service.create(fileId=file_id, body=permission_writer, sendNotificationEmail=False).execute()
        self._permission_service.create(fileId=file_id, body=permission_reader, sendNotificationEmail=False).execute()
    ```

### Usage

```python
helper = GoogleDriveHelper(
  project_root_dir=root_dir, # 프로젝트의 루트 경로입니다.
  google_client_key_path=google_client_key_path, # GCP에서 만든 google service account의 key가 있는 경로입니다.
  google_drive_owner_email=google_drive_owner_email, # 권한 이전용 이메일입니다.
  do_not_download_list=do_not_download_list, # 다운로드하지 않을 파일 및 폴더 이름을 적으면 됩니다.
  local_target_root_dir_name="model", # 업로드할 파일들이 모여있는 로컬 디렉터리입니다.
  drive_root_folder_name='comment-filtering', # google drive에서 사용할 root folder 이름입니다.
  test_mode=True # 테스트 모드를 켭니다.
)

# 메타데이터 관련
#   구글 드라이브의 디렉터리 경로를 dictionary로 초기화합니다.
helper.create_folder_and_file_metadatas()
helper.print_directory_metadata()

# 파일이나 폴더의 유무 확인
#   folder_name: 폴더 이름입니다. 필수 항목입니다.
#   filename: 파일 이름입니다. 만약 없다면 폴더를 검색하고 있다면 파일을 검색합니다.
helper.is_exists(folder_name, filename)

# 파일 업로드
#   초기화시 설정한 local_target_root_dir_name 밑에 있는 모든 파일을 업로드합니다.
helper.upload_all_files()
#   특정 파일을 업로드합니다.
#     file_path: 파일의 경로입니다.
#     parent_filder_name: google drive의
#     additional_path:
helper.upload_file(file_path, parent_folder_name, additional_path)

# 파일 다운로드
#   초기화시 설정한 drive_root_folder_name 밑의 모든 파일을 다운로드 합니다.
helper.download_all_files()
#   특정 파일을 다운로드합니다. 같은 이름이 여러 개 존재한다면 디렉터리를 만들고 다운로드 합니다.
#     filename: 파일의 이름입니다.
helper.download_file(filename)

# 파일 삭제
#   파일을 완전 삭제합니다.
helper.delete_file(file_id)
#   파일을 휴지통으로 이동시킵니다.
helper.move_to_trash(file_id)
```
