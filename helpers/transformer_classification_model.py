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
    def __init__(self, model_type:str, base_model_path:str="./model", quantize:str='fp32'):
        self.model_type=model_type
        self.quantize = quantize

        self.model_path=os.path.join(base_model_path, f"{model_type}_model" + ('_fp16' if quantize == 'fp16' else ''))
        self.tokenizer_path=os.path.join(base_model_path, f"tokenizer")
        self.dataset_path=os.path.join(base_model_path, 'dataset.csv')

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.max_token_length=256 if model_type == "comment" else 40

        self.model=None

    def load(self):
        try:
            if self.model is not None:
                return
            
            self.generate_label_array()
            
            self._raise_file_not_found(self.model_path)
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
        self._raise_file_not_found(self.dataset_path)
        target_column = f"{self.model_type}_class"

        data=pd.read_csv(self.dataset_path, 
                         usecols=[target_column],
                         dtype={target_column: str})
        self.label_array=data[target_column].dropna().unique().tolist()

    def _raise_file_not_found(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not found at {path}")
    
    def predict(self, texts: Union[str, List[str]]) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.padding_side = 'right'
        
        if isinstance(texts, str):
            texts = [texts]

        batch_size = 16 * 2 if self.quantize == 'fp16' else 1

        predict_as_long_text_border_length = 300
        short_texts = [[idx, text] for idx, text in enumerate(texts) if len(text) <= predict_as_long_text_border_length]
        long_texts = [[idx, text] for idx, text in enumerate(texts) if len(text) > predict_as_long_text_border_length]

        all_predictions = [None] * len(texts)

        def predict_short_texts(texts: List[Union[int, str]]):
            origin_indicies, batch_texts = zip(*texts)
            tokens = tokenizer(batch_texts,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length = self.max_token_length)
            
            model_inputs = {key: val.to(self.device) for key, val in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            probs = F.softmax(outputs.logits, dim=1)
            predict_class_indicies = torch.argmax(outputs.logits, dim=1).tolist()

            for origin_index, predict_index, prob in zip(origin_indicies, predict_class_indicies, probs):
                all_predictions[origin_index] = (self.label_array[predict_index], [ round(p, 2) for p in prob.tolist() ])
        
        # 짧은 문장은 배치처리하여 추론
        # 짧은 문장도 문장별로 추론해야할까?
        for i in range(0, len(short_texts), batch_size):
            predict_short_texts(short_texts[i:i+batch_size])
            if (i+1) % (4*batch_size) == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()


        def predict_long_texts(inputs, origin_indicies):
            def pad_tensor(tensors: List[DefaultDict[str, List[torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                temp_max_length = max(len(d_dict[key]) for d_dict in tensors for key in d_dict.keys())
                return {
                    key: torch.stack([
                        F.pad(tensor, (0, temp_max_length - len(tensor)), mode='constant', value=tokenizer.pad_token_id if key == 'input_ids' else 0)
                        for d_dict in tensors for tensor in d_dict[key]
                    ]) for key in tensors[0].keys()
                }

            padded = pad_tensor(inputs)
            model_inputs = { key: val.to(self.device) for key, val in padded.items() }

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            probs = F.softmax(outputs.logits, dim=1)
            predicted_class_indicies = torch.argmax(outputs.logits, dim=1).tolist()
            
            for origin_idx, pred_class_idx, prob in zip(origin_indicies, predicted_class_indicies, probs):
                grouped_data[origin_idx].append([pred_class_idx, prob])

        # 토큰 제한이 있다. 512개. 따라서 더 줄이는 방법이 필요...
        # .?!^ 단위로 나누고 이걸 토크나이징 하는게 나을지도...?
        def calculate_token_length(text: str) -> Tuple[DefaultDict[str, List[torch.Tensor]], List[int]]:
            token_collapse_length = 20

            splitted_text = re.split(r'([.,!?\^~]{2,}|[.!?~])(?=\s)', text)
            merged_sentences = [splitted_text[i].strip() + splitted_text[i+1].strip()
                                for i in range(0, len(splitted_text)-1, 2)]
            if len(splitted_text) % 2 != 0:
                merged_sentences.append(splitted_text[-1].strip())

            tensor_result = defaultdict(list)
            sentence_lengths = []

            for sentence in merged_sentences:
                tokens = tokenizer(sentence, return_tensors="pt")
                token_length = tokens['input_ids'][0].size(0)

                cur_token_idx = 0
                while cur_token_idx < token_length:
                    # 두 값의 차이가 {{ token_collapse_length }} 초과라면 현재 문장이 유의미한 문장임을 암시
                    # 그것이 아니라면 유의미하지 않음 -> 유의미한 전 토큰을 가져와서 토큰화
                    if (token_length - cur_token_idx) > token_collapse_length:
                        start_idx = cur_token_idx
                        end_idx = min(cur_token_idx + self.max_token_length, token_length)
                        cur_token_idx = cur_token_idx + self.max_token_length - token_collapse_length
                    else:
                        start_idx = token_length - self.max_token_length
                        end_idx = token_length
                        cur_token_idx = token_length # 종료조건 활성화

                    local_result = {key: tokens[key][0][start_idx:end_idx] for key in tokens}
                    batched_token_length = [len(local_result[key]) for key in tokens][0]
                
                    sentence_lengths.append(batched_token_length)
                    for key in tokens:
                        tensor_result[key].append(local_result[key])

            return tensor_result, sentence_lengths

        # 긴 문장은 토큰별로 추론
        tensor_counter = 0
        inputs = []
        origin_indicies = []
        grouped_data = defaultdict(list)
        for loop_idx, value in enumerate(long_texts):
            origin_idx, text = value
            tensors, tensor_lengths = calculate_token_length(text)
            ## tensor_lengths
            #### [47, 26, 47, 26, 47, 26, 47, 26, 47, 26, 47, 26]
            #### 각 tensor 토큰의 길이를 반환. 가중치를 이걸 이용해서 계산하려고 했는데... 조금 힘들것같다
            
            inputs.append(tensors)
            origin_indicies.extend([origin_idx] * len(tensor_lengths))
            tensor_counter = tensor_counter + len(tensor_lengths)

            if tensor_counter >= batch_size:
                predict_long_texts(inputs, origin_indicies)

                inputs.clear()
                origin_indicies.clear()
                tensor_counter = 0

            if (loop_idx+1) % (4*batch_size) == 0:
                torch.cuda.empty_cache()

        if len(inputs) != 0:
            predict_long_texts(inputs, origin_indicies)
            torch.cuda.empty_cache()

        for key, value in grouped_data.items():
            probs = torch.stack([ prob for _, prob in value ])
            probs_mean = probs.mean(dim=0)

            max_prob_index = probs_mean.argmax()
            rounded_probs = (probs_mean * 100).round() / 100

            all_predictions[key] = (self.label_array[max_prob_index], 
                                    rounded_probs.tolist())
            
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

    