from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn.functional as F

import gc
import os
import pandas as pd
import re

from typing import Any, List, Union, Tuple, Dict, DefaultDict
from collections import defaultdict

class TransformerClassificationModel:
    def __init__(self, model_type:str, base_model_path:str="./model", quantize:str='fp32'):
        self.model_type = model_type
        self.quantize = quantize

        self.model_path = os.path.join(base_model_path, f"{model_type}_model" + ('_fp16' if quantize == 'fp16' else ''))
        self.tokenizer_path = os.path.join(base_model_path, f"tokenizer")
        self.dataset_path = os.path.join(base_model_path, 'dataset.csv')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.max_token_length = 256 if model_type == "comment" else 40

        self.model = None # 없으면 load에서 뭐라 뭐라 한다

    def load(self):
        try:
            if self.model is not None:
                return
            self.label_array = self.generate_label_array()

            self._raise_file_not_found(self.tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            tokenizer.padding_side = 'right'
            self.tokenizer = tokenizer
            
            self._raise_file_not_found(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            model.to(self.device)
            model.eval()

            self.model = model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def unload(self):
        try:
            if self.model:
                del self.model, self.tokenizer, self.label_array
                self.model = None
                self.tokenizer = None
                self.label_array = None
            gc.collect()
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
    
    def reload(self):
        self.unload()
        self.load()

    def generate_label_array(self) -> List[str]:
        self._raise_file_not_found(self.dataset_path)
        target_column = f"{self.model_type}_class"

        data = pd.read_csv(
            self.dataset_path, 
            usecols=[target_column],
            dtype={target_column: str}
        )[target_column]
        return data.drop_duplicates().dropna().tolist()

    def _raise_file_not_found(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"file not found at {path}")
        
    def is_long_text(self, text: str) -> bool:
        predict_as_long_text_border_length = int(self.max_token_length * 1.1)
        total_length = len(text)
        for matched_token in re.findall(r'\[[A-Z_]+\]|\s', text):
            total_length = total_length - len(matched_token)
        
        return total_length >= predict_as_long_text_border_length

    def predict_short_texts(self, texts: List[Union[int, str]], all_predictions: List[Any]):
        origin_indicies, batch_texts = zip(*texts)
        tokens = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length = self.max_token_length
        )
        
        model_inputs = {key: val.to(self.device) for key, val in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        probs = F.softmax(outputs.logits, dim=1)
        predict_class_indicies = torch.argmax(outputs.logits, dim=1).tolist()

        for origin_index, predict_index, prob in zip(origin_indicies, predict_class_indicies, probs):
            all_predictions[origin_index] = (
                self.label_array[predict_index],
                [ round(p, 2) for p in prob.tolist() ]
            )

    def predict_long_texts(self, tensors: List[DefaultDict[str, List[torch.Tensor]]], origin_indicies: List[int], grouped_data: DefaultDict[str, List[any]]):
        local_max_length = max(len(tensor) for d_dict in tensors for tensor in d_dict.get('input_ids', []))
        padded = {
            key: torch.stack([
                F.pad(tensor, (0, local_max_length - len(tensor)), mode='constant', value=self.tokenizer.pad_token_id if key == 'input_ids' else 0)
                for d_dict in tensors for tensor in d_dict[key]
            ]) for key in tensors[0].keys()
        }
        model_inputs = { key: val.to(self.device) for key, val in padded.items() }

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        probs = F.softmax(outputs.logits, dim=1)
        predicted_class_indicies = torch.argmax(outputs.logits, dim=1).tolist()
        
        for origin_idx, pred_class_idx, prob in zip(origin_indicies, predicted_class_indicies, probs):
            grouped_data[origin_idx].append([pred_class_idx, prob])

    # 토큰 제한이 있다. 512개. 따라서 더 줄이는 방법이 필요...
    # .?!^ 단위로 나누고 이걸 토크나이징 하는게 나을지도...?
    def calculate_token_length(self, text: str) -> Tuple[DefaultDict[str, List[torch.Tensor]], List[int]]:
        token_collapse_length = 20

        splitted_text = re.split(r'([.,!?\^~]{1,})', text)
        merged_sentences = [splitted_text[i].strip() + splitted_text[i+1].strip()
                            for i in range(0, len(splitted_text)-1, 2)]
        if len(splitted_text) % 2 != 0:
            merged_sentences.append(splitted_text[-1].strip())

        tensor_result = defaultdict(list)
        sentence_lengths = []

        for sentence in merged_sentences:
            tokens = self.tokenizer(sentence, return_tensors="pt")
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
    
    def predict(self, texts: Union[str, List[str]]) -> List[Tuple[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        
        if isinstance(texts, str):
            texts = [texts]

        batch_size = 16 * (2 if self.quantize == 'fp16' else 1)

        short_texts, long_texts = [], []
        for idx, text in enumerate(texts):
            (long_texts if self.is_long_text(text) else short_texts).append([idx, text])

        all_predictions = [None] * len(texts)
        
        # 짧은 문장은 배치처리하여 추론
        # 짧은 문장도 문장별로 추론해야할까?
        for i in range(0, len(short_texts), batch_size):
            self.predict_short_texts(short_texts[i:i+batch_size], all_predictions)
            if (i+1) % (4*batch_size) == 0:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # 긴 문장은 토큰별로 추론
        tensor_counter = 0
        batch_inputs = []
        origin_indicies = []
        grouped_data = defaultdict(list)
        for loop_idx, long_text_iter in enumerate(long_texts):
            origin_idx, text = long_text_iter
            tensors, tensor_lengths = self.calculate_token_length(text)
            ## tensor_lengths
            #### [47, 26, 47, 26, 47, 26, 47, 26, 47, 26, 47, 26]
            #### 각 tensor 토큰의 길이를 반환. 가중치를 이걸 이용해서 계산하려고 했는데... 조금 힘들것같다
            
            batch_inputs.append(tensors)
            origin_indicies.extend([origin_idx] * len(tensor_lengths))
            tensor_counter = tensor_counter + len(tensor_lengths)

            if tensor_counter >= batch_size:
                self.predict_long_texts(batch_inputs, origin_indicies, grouped_data)

                batch_inputs.clear()
                origin_indicies.clear()
                tensor_counter = 0

            if (loop_idx+1) % (4*batch_size) == 0:
                torch.cuda.empty_cache()

        if tensor_counter != 0:
            self.predict_long_texts(batch_inputs, origin_indicies, grouped_data)
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

    