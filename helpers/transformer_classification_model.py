from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn.functional as F

import gc
import os
import pandas as pd
import re

from typing import List, Union, Tuple, Dict, DefaultDict
from collections import defaultdict

class TransformerClassificationModel:
    def __init__(self, model_type:str, base_model_path:str="./model"):
        self.model_type=model_type

        self.model_path=os.path.join(base_model_path, f"{model_type}_model")
        self.tokenizer_path=os.path.join(base_model_path, f"tokenizer")
        self.dataset_path=os.path.join(base_model_path, 'dataset.csv')

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.max_token_length=256 if model_type == "comment" else 40

        self.model=None

    def load(self):
        try:
            if self.model:
                return
            
            self.generate_label_array()
            
            self._raise_file_not_fount(self.model_path)
            self.model=AutoModelForSequenceClassification.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def unload(self):
        try:
            if self.model:
                self.model=None
            gc.collect()
        
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
    
    def reload(self):
        self.unload()
        self.load()

    def generate_label_array(self):
        self._raise_file_not_fount(self.dataset_path)

        data=pd.read_csv(self.dataset_path, usecols=[f"{self.model_type}_class"])
        self.label_array=data[f"{self.model_type}_class"].dropna().unique()

    def _raise_file_not_fount(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not found at {path}")
    
    def predict(self, texts: Union[str, List[str]]) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        if isinstance(texts, str):
            texts = [texts]

        batch_size = 16

        short_texts = []
        long_texts = []

        predict_as_long_text_border_length = 300

        for idx, text in enumerate(texts):
            if len(text) <= predict_as_long_text_border_length:
                short_texts.append([idx,text])
            else:
                long_texts.append([idx,text])

        all_predictions = [None] * len(texts)

        def predict_short_texts(texts: List[Union[int, str]]):
            origin_indicies, batch_texts = zip(*texts)
            tokens = tokenizer(list(batch_texts),
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    padding_side="right",
                                    max_length = self.max_token_length)
            model_inputs = {key: val.to(self.device) for key, val in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            probs = F.softmax(outputs.logits, dim=1)
            predict_class_indicies = torch.argmax(outputs.logits, dim=1).tolist()

            for origin_index, predict_index, prob in zip(origin_indicies, predict_class_indicies, probs):
                all_predictions[origin_index] = (self.label_array[predict_index], [ round(p, 2) for p in prob.tolist() ])

            torch.cuda.empty_cache()
        
        # 짧은 문장은 배치처리하여 추론
        # 짧은 문장도 문장별로 추론해야할까?
        for i in range(0, len(short_texts), batch_size):
            predict_short_texts(short_texts[i:i+batch_size])
            if (i+1) % (4*batch_size) == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()


        def predict_long_texts(inputs, local_indicies):
            def pad_tensor(tensors: List[DefaultDict[str, List[torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                keys = list(tensors[0].keys())
                max_length = max(
                    max(len(tensor) for tensor in d_dict[keys[0]]) 
                    for d_dict in tensors
                )
                
                # tensor.pad에서, 1차원 tensor의 경우에는 (left_pad, right_pad)를 의미한다.
                # 즉 left_pad는 왼쪽에 추가할 개수, right_pad는 오른쪽에 추가할 개수이다.
                return {
                    key: torch.stack([
                            F.pad(tensor, (0, max_length - len(tensor)), mode='constant', value=tokenizer.pad_token_id if key == 'input_ids' else 0) for d_dict in tensors for tensor in d_dict[key]
                        ]) for key in keys
                }

            padded = pad_tensor(inputs)
            model_inputs = { key: val.to(self.device) for key, val in padded.items() }

            with torch.no_grad():
                outputs = self.model(**model_inputs)
                probs = F.softmax(outputs.logits, dim=1)
                predicted_class_indicies = torch.argmax(outputs.logits, dim=1).tolist()
            
            for local_idx, pred_idx, prob in zip(local_indicies, predicted_class_indicies, probs):
                grouped_data[local_idx].append([pred_idx, prob])

        # 토큰 제한이 있다. 512개. 따라서 더 줄이는 방법이 필요...
        # .?!^ 단위로 나누고 이걸 토크나이징 하는게 나을지도...?
        def calculate_token_length(text: str):
            def process_token(tokens, start_index: int, end_index: int):
                result = defaultdict(list)
                batched_length = []
                for key in tokens:
                    batched = tokens[key][0][start_index:end_index]
                    batched_length.append(len(batched))
                    result[key].append(batched)

                return result, batched_length

            token_collapse_length = 20

            splitted_text = re.split(r'([.,!?^~]{2,}|[.!?~])(?=\s)', text)
            merged_sentences = []
            for i in range(0, len(splitted_text)-1, 2):
                merged_sentences.append(splitted_text[i].strip() + splitted_text[i+1].strip())

            if len(splitted_text) % 2 != 0:
                merged_sentences.append(splitted_text[-1].strip())

            tensor_result = defaultdict(list)
            sentence_lengths = []

            for sentence in merged_sentences:
                tokens = tokenizer(sentence, return_tensors="pt")
                token_length = len(tokens['input_ids'][0])

                cur_token_idx = 0
                while cur_token_idx < token_length:
                    # 두 값의 차이가 {{ token_collapse_length }} 초과라면 현재 문장이 유의미한 문장임을 암시
                    # 그것이 아니라면 유의미하지 않음 -> 유의미한 전 토큰을 가져와서 토큰화
                    if (token_length - cur_token_idx) > token_collapse_length:
                        local_result, batched_lengths = process_token(tokens, 
                                                                      cur_token_idx,
                                                                      cur_token_idx + self.max_token_length)
                        cur_token_idx = cur_token_idx + self.max_token_length - token_collapse_length
                    else:
                        local_result, batched_lengths = process_token(tokens, 
                                                                      token_length - self.max_token_length,
                                                                      self.max_token_length)
                        cur_token_idx = token_length # 종료조건 활성화
                
                    sentence_lengths.append(batched_lengths)
                    for key in tokens:
                        tensor_result[key].extend(local_result[key])

            return tensor_result, sentence_lengths

        # 긴 문장은 토큰별로 추론
        local_sum = 0
        inputs = []
        local_indicies = []
        grouped_data = defaultdict(list)
        for loop_idx, value in enumerate(long_texts):
            idx, text = value
            tensors, tensor_lengths = calculate_token_length(text)
            
            inputs.append(tensors)
            local_indicies.extend([idx] * len(tensor_lengths))
            local_sum = local_sum + len(tensor_lengths)

            if local_sum > batch_size:
                predict_long_texts(inputs, local_indicies)

                inputs = []
                local_indicies = []
                local_sum = 0

            if (loop_idx+1) % (4*batch_size) == 0:
                torch.cuda.empty_cache()

        if len(inputs) != 0:
            predict_long_texts(inputs, local_indicies)
            torch.cuda.empty_cache()

        for key, value in grouped_data.items():
            probs = torch.stack([ prob for _, prob in value ])
            probs_mean = probs.mean(dim=0)

            all_predictions[key] = (
                self.label_array[probs_mean.argmax()], 
                [ round(p, 2) for p in probs_mean.tolist() ]
            )
            
        return all_predictions, self.label_array

if __name__ == "__main__":
    print('cuda available: ', torch.cuda.is_available())

    root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comment_model=TransformerClassificationModel(model_type="comment",
                                                  base_model_path=root_dir+'/model')
    comment_model.load()

    text=input("검증할 댓글을 입력해주세요: ")
    predicted=comment_model.predict([text])
    print(predicted, type(predicted))

    