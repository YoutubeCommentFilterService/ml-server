try:
    from transformers import AutoTokenizer
except:
    from tokenizers import Tokenizer

import onnxruntime as ort

import gc
import os
import pandas as pd
import numpy as np
import re

from typing import Union, Optional, List

class ONNXClassificationModel:
    def __init__(self, model_type:str, base_path:str="./model"):
        self.model_type=model_type

        self.onnx_model_path=os.path.join(base_path, f"{model_type}_onnx", 'model.onnx')
        self.tokenizer_path=os.path.join(base_path, f"{model_type}_tokenizer")
        self.dataset_path=os.path.join(base_path, 'dataset.csv')

        self.max_length=256 if self.model_type == "comment" else 40

        self.session=None
        self.tokenizer=None

    def load(self):
        try:
            if self.session:
                return
            
            self.generate_label_array()

            self._raise_file_not_fount(self.tokenizer_path)
            if os.getenv("HARDWARE") == "JETSON_NANO":
                self.tokenizer = Tokenizer.from_file(f"{self.tokenizer_path}/tokenizer.json")
            else:
                self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)

            self._raise_file_not_fount(self.onnx_model_path)
            sess_options=ort.SessionOptions()
            sess_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads=0 # 연산 내부를 병렬화, 0은 cpu 코어 개수를 알아서 설정
            sess_options.inter_op_num_threads=1 # 연산 간 병령 실행, NLP는 대부분 직렬

            sess_options.enable_mem_pattern = True  # 8GB RAM이므로 활성화

            # 병렬 실행 최적화. ORT_SEQUENTIAL / ORT_PARALLEL
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                providers = ['CPUExecutionProvider']

            # ONNX 세션 생성
            self.session=ort.InferenceSession(self.onnx_model_path,
                                              sess_options=sess_options,
                                              providers=providers)

            self.input_names=[input.name for input in self.session.get_inputs()]

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def unload(self):
        try:
            if self.session:
                del self.session
                self.session=None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer=None

            if self.input_names:
                del self.input_names
                self.input_names=None

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
    
    def predict(self, texts: Union[str, List[str]]):
        if self.session is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")
        
        try:
            if isinstance(texts, str):
                texts = [texts]

            inputs = self._generate_inputs(texts)


            # if os.getenv("HARDWARE") == "JETSON_NANO":
            #     encoded_batch = self.tokenizer.encode_batch(texts)

            #     input_ids = np.array([enc.ids[:self.max_length] + [0] * (self.max_length - len(enc.ids)) for enc in encoded_batch], dtype=np.int64)
            #     attention_mask = np.array([[1] * min(len(enc.ids), self.max_length) + [0] * (self.max_length - len(enc.ids)) for enc in encoded_batch], dtype=np.int64)
            #     token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            #     inputs = {
            #         "input_ids": input_ids,
            #         "attention_mask": attention_mask,
            #         "token_type_ids": token_type_ids
            #     }

            # else:
            #     inputs = self.tokenizer(texts,
            #                             padding="max_length",
            #                             truncation=True,
            #                             max_length=self.max_length,
            #                             return_tensors="np")
            #     inputs={name: inputs[name].astype(np.int64) for name in self.input_names}

            output_values = []

            batch_size = 64
            for i in range(0, len(texts), batch_size):
                input_batch = {name: inputs[name][i:i+batch_size].astype(np.int64) for name in self.input_names}
                outputs = self.session.run(None, input_batch)
                output_values.extend(outputs[0])
                del outputs, input_batch

            predicted_class_indeces = np.argmax(output_values, axis=-1)
            predicted_labels = [self.label_array[idx] for idx in predicted_class_indeces]

            del inputs
            gc.collect()
            
            return predicted_labels
        
        except Exception as e:
            print(e)
            return []

    def _generate_inputs(self, texts: List[str]):
        if os.getenv("HARDWARE") == "JETSON_NANO":
            encoded_batch = self.tokenizer.encode_batch(texts)

            input_ids = np.array([enc.ids[:self.max_length] + [0] * (self.max_length - len(enc.ids)) for enc in encoded_batch], dtype=np.int64)
            attention_mask = np.array([[1] * min(len(enc.ids), self.max_length) + [0] * (self.max_length - len(enc.ids)) for enc in encoded_batch], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }

        else:
            inputs = self.tokenizer(texts,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors="np")
            inputs={name: inputs[name].astype(np.int64) for name in self.input_names}

        return inputs

    
if __name__ == "__main__":
    root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nickname_model=ONNXClassificationModel(model_type="nickname",
                                           base_path=root_dir+'/model')
    nickname_model.load()
    
    text=input("검증할 닉네임을 입력해주세요: ")
    nickname=re.sub(r"-._", " ", text)
    print(nickname_model.predict(text))