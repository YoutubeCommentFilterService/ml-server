from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import pandas as pd

import re
import os
import traceback
import unicodedata

import torch
from helpers import TransformerClassificationModel, GoogleDriveHelper, Tokenizer

do_not_download_list = ['dataset-backup']

project_root_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(project_root_dir, "env", ".env"))
google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")
do_not_download_list = ['dataset-backup']
google_client_key_path = os.path.join(project_root_dir, 'env', 'ml-server-key.json')

helper = GoogleDriveHelper(project_root_dir=project_root_dir,
                           google_client_key_path=google_client_key_path,
                           google_drive_owner_email=google_drive_owner_email,
                           do_not_download_list=do_not_download_list,
                           local_target_root_dir_name='model',
                           drive_root_folder_name='comment-filtering')
if not os.path.exists('./model'):
    helper.download_all_files()

try:
    tokenizer = Tokenizer().get_tokenizer()
except ImportError:
    raise ImportError('plz install transformers package')

if torch.cuda.is_available():
    comment_model = TransformerClassificationModel(model_type="comment")
    nickname_model = TransformerClassificationModel(model_type="nickname")

    do_not_download_list.extend(['comment_onnx', 'nickname_onnx'])

    print("Transformer loaded")

    model_type = 'transformer'
else:
    raise ValueError("CUDA not available")

app = FastAPI()

class PredictItem(BaseModel):
    id: str
    nickname: str
    comment: str
    
class PredictResult(BaseModel):
    id: str
    nickname_predicted: str
    nickname_predicted_prob: List[float]
    comment_predicted: str
    comment_predicted_prob: List[float]

class PredictRequest(BaseModel):
    items: List[PredictItem]

class PredictResponse(BaseModel):
    items: List[PredictResult]
    model_type: Optional[str]
    nickname_categories: List[str]
    comment_categories: List[str]

async def startup():
    nickname_model.load(tokenizer=tokenizer)
    comment_model.load(tokenizer=tokenizer)

async def shutdown():
    nickname_model.unload()
    comment_model.unload()

def normalize_unicode_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    unicode_single_hangul_dict = {'ᄀ': 'ㄱ', 'ᄂ': 'ㄴ', 'ᄃ': 'ㄷ', 'ᄅ': 'ㄹ', 'ᄆ': 'ㅁ', 'ᄇ': 'ㅂ', 'ᄉ': 'ㅅ', 'ᄋ': 'ㅇ', 'ᄌ': 'ㅈ', 'ᄎ': 'ㅊ', 'ᄏ': 'ㅋ', 'ᄐ': 'ㅌ', 'ᄑ': 'ㅍ', 'ᄒ': 'ㅎ', 'ᄍ': 'ㅉ', 'ᄄ': 'ㄸ', 'ᄁ': 'ㄲ', 'ᄊ': 'ㅆ', 'ᅡ': 'ㅏ', 'ᅣ': 'ㅑ', 'ᅥ': 'ㅓ', 'ᅧ': 'ㅕ', 'ᅩ': 'ㅗ', 'ᅭ': 'ㅛ', 'ᅮ': 'ㅜ', 'ᅲ': 'ㅠ', 'ᅳ': 'ㅡ', 'ᅵ': 'ㅣ', 'ᅢ': 'ㅐ', 'ᅦ': 'ㅔ', 'ᅴ': 'ㅢ', 'ᆪ': 'ㄱㅅ', 'ᆬ': 'ㄴㅈ', 'ᆭ': 'ㄴㅎ', 'ᆲ': 'ㄹㅂ', 'ᆰ': 'ㄹㄱ', 'ᆳ': 'ㄹㅅ', 'ᆱ': 'ㄹㅁ', 'ᄚ': 'ㄹㅎ', 'ᆴ': 'ㄹㅌ', 'ᆵ': 'ㄹㅍ', 'ᄡ': 'ㅂㅅ', 'ᄈ': 'ㅂㅂ'}
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))

    return ''.join(unicode_single_hangul_dict[ch] if ch in unicode_single_hangul_dict else ch for ch in normalized)

app.add_event_handler("startup", startup)
app.add_event_handler("shutdown", shutdown)
    
import time

from typing import Union
def normalize_tlettak_font(text: str, 
                           space_pattern: Union[str, re.Pattern] = r'[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]+[\s!?@.,❤]*', 
                           search_pattern: Union[str, re.Pattern] = r'(\b\w\b)([\s!?@.,❤]+)(\b\w\b)',
                           ) -> str:
    if isinstance(space_pattern, str):
        space_pattern = re.compile(space_pattern)
    if isinstance(search_pattern, str):
        search_pattern = re.compile(search_pattern)

    result = []
    sub = []
    pos = 0
    
    while pos < len(text):
        space_matched = space_pattern.match(text, pos)
        search_matched = search_pattern.match(text, pos)

        if search_matched:
            sub.extend([search_matched.group(1), search_matched.group(3)])
            pos = search_matched.end() - 1
        elif space_matched:
            s_end = space_matched.end()
            result.append(''.join(sub[::2]) + text[pos:s_end].strip())
            pos = s_end
            sub.clear()
        else:   # 둘 다 매칭 실패인 경우 뒷문장 전부를 붙여씀
            result.append(text[pos:])
            break
    return ' ' .join(result)

pattern_spacer = '=!?@'
space_pattern = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9:]+[\s!?@.,❤]*')
pattern = re.compile(rf"[{pattern_spacer}]*(\w)([{pattern_spacer}\s.,❤]+)(\w)")

def replace_regex_predict_data(df: pd.DataFrame):
    # prefix, subfix 제거
    df['nickname'] = df['nickname']\
        .str.strip()\
        .str.replace('@', '')\
        .str.replace(r'-[a-zA-Z0-9]+(?=\s|$)', '', regex=True)
    # 특수 기호 제거
    df['nickname'] = df['nickname']\
        .str.replace(r'[-._]', '', regex=True)
    # 영어, 한글, 숫자가 아닌 경우 기본 닉네임 처리
    df['nickname'] = df['nickname']\
        .str.replace(r'[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9]+', '[DEFAULT_NICK]', regex=True)
    
    with open('./tokens/emojis.txt', 'r', encoding='utf-8') as f:
        emojis = [line.strip() for line in f.readlines()]

    emoji_pattern = '|'.join(map(re.escape, emojis))
    df['comment'].str.replace(emoji_pattern, '[TEXT_EMOJI]', regex=True)
    
    # 유니코드 문장부호 수정
    df['comment'] = df['comment']\
        .str.replace(r'[ㆍ·・•]', '.', regex=True)\
        .str.replace(r'[ᆢ…]+', '..', regex=True)\
        .str.replace(r'[‘’]+', "'", regex=True)\
        .str.replace(r'[“”]+', '"', regex=True)\
        .str.replace(r'[\u0020\u200b\u2002\u2003\u2007\u2008\u200c\u200d]+', ' ', regex=True)\
        .str.replace(r'[\U0001F3FB-\U0001F3FF\uFE0F]', '', regex=True)
    # 유니코드 꾸밈 문자(결합 문자) 제거
    df['comment'] = df['comment'].str.replace(r'\*+', '', regex=True)
    df['comment'] = df['comment'].apply(lambda x: normalize_unicode_text(x) if isinstance(x, str) else x)
    # special token 파싱
    df['comment'] = df['comment']\
        .str.replace(r'https?:\/\/(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ-]+\.[a-zA-Z]{2,}(?:\/[^?\s]*)?(?:\?[^\s]*)?', '[URL]', regex=True)\
        .str.replace(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', regex=True)\
    # 한글자 + 부호 + 한글자 패턴 처리
    df['comment'] = df['comment'].apply(lambda x: normalize_tlettak_font(x, space_pattern, pattern) if isinstance(x, str) else x)
    # special token 파싱
    df['comment'] = df['comment']\
        .str.replace(r'@{1,2}[A-Za-z0-9가-힣\_\-\.]+', '[TAG]', regex=True)\
        .str.replace(r'#[A-Za-z0-9ㄱ-ㅎㅏ-ㅣ가-힣\_\-\.]+', '[HASH_TAG]', regex=True)\
        .str.replace(r'([👇✋👍])', '[THUMB]', regex=True)\
        .str.replace(r'([➡⬇↗↘↖↙→←↑↓⇒]|[\-\=]+>|<[\-\=]+)', '[ARROW]', regex=True)
    # 쓸데없이 많은 문장부호 제거
    df['comment'] = df['comment']\
        .str.replace(r'([^\s])[.,](?=\S)', r'\1', regex=True)\
        .str.replace(r'([.,?!^]+)', r' \1 ', regex=True)\
        .str.replace(r'\s+([.,?!^]+)', r'\1', regex=True)\
        .str.replace(r'\s{2,}', ' ', regex=True)
    # timestamp 처리
    to_replace = '[TIMESTAMP]'
    df['comment'] = df['comment']\
        .str.replace(r'\d+:(?:\d+:?)?\d+', to_replace, regex=True)
    # 밈 처리
    # df['comment'] = df['comment']\
    #     .str.replace(r'(?i)chill', '칠', regex=True)
    # 한글, 영어가 아닌 경우 처리
    df['comment'] = df['comment']\
        .str.replace(r'[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ♡♥\!\?\@\#\$\%\^\&\*\(\)\-\_\=\+\\\~\,\.\/\<\>\[\]\{\}\;\:\'\"\s]', '', regex=True)
    # 2개 이상 연속된 문자 처리
    df['comment'] = df['comment']\
        .str.replace(r'(.)\1{2,}', r'\1\1', regex=True)
    # 빈 문자열의 경우 empty 처리
    df['comment'] = df['comment'].str.strip()
    df['comment'] = df['comment'].fillna('[EMPTY]')

@app.post("/predict")
def predict_batch(data: PredictRequest):
    print('predict request accepted...')
    try:
        items = data.items
        response_data = []

        df = pd.DataFrame([{'nickname': item.nickname, 'comment': item.comment} for item in items])
        replace_regex_predict_data(df)

        nicknames = df['nickname'].tolist()
        comments = df['comment'].tolist()

        start = time.time()
        nickname_outputs, nickname_categories = nickname_model.predict(nicknames)
        comment_outputs, comment_categories = comment_model.predict(comments)
        print(f"predict len: {len(items)}, time: {time.time() - start}")

        print(len(items), len(comment_outputs), len(nickname_outputs))

        index = 0
        for item, comment_output, nickname_output in zip(items, comment_outputs, nickname_outputs):
            if nickname_output is None:
                print(f'\tnickname{index} = {nickname_output}, {item}')
            if comment_output is None:
                print(f'\tcomment{index} = {comment_output}, {item}')
            index = index + 1
            
            response_data.append(PredictResult(id=item.id, 
                                               nickname_predicted=nickname_output[0],
                                               nickname_predicted_prob=nickname_output[1],
                                               comment_predicted=comment_output[0],
                                               comment_predicted_prob=comment_output[1]))
        
        # 결과 반환
        return PredictResponse(items=response_data, model_type=model_type, nickname_categories=nickname_categories, comment_categories=comment_categories)

    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.patch("/update")
def update_dataset():
    try:
        helper.download_all_files()
        tokenizer = Tokenizer().reload_tokenizer()
        nickname_model.reload(tokenizer)
        comment_model.reload(tokenizer)
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
