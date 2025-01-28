from transformers import AutoTokenizer
import onnxruntime

import gc
import os
import pandas as pd
import numpy as np
import re

class ONNXClassificationModel:
    def __init__(self, model_type:str, base_path:str="./model"):
        self.model_type=model_type

        self.quantize_path=os.path.join(base_path, f"{model_type}_quantize", 'model_quantized.onnx')
        self.tokenizer_path=os.path.join(base_path, f"{model_type}_tokenizer")
        self.dataset_path=os.path.join(base_path, 'dataset.csv')

        self.max_length=256 if self.model_type == "comment" else 40

        self.session=None
        self.tokenizer=None

    def load(self):
        try:
            self.generate_label_array()

            self._raise_file_not_fount(self.tokenizer_path)
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)

            self._raise_file_not_fount(self.quantize_path)
            sess_options=onnxruntime.SessionOptions()
            sess_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads=1

            # ONNX 세션 생성
            providers=['CPUExecutionProvider']
            self.session=onnxruntime.InferenceSession(self.quantize_path,
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

    def predict(self, text):
        if self.session is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")
        
        # 토크나이저를 사용하여 입력 텍스트 전처리
        inputs=self.tokenizer(text,
                              padding=True,
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="np")

        inputs={name: inputs[name].astype(np.int64) for name in self.input_names}

        outputs=self.session.run(None, inputs)
        output_values=outputs[0]
        predicted_class_index=np.argmax(output_values, axis=-1).item()

        del inputs, outputs

        return self.label_array[predicted_class_index]
    
if __name__ == "__main__":
    root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nickname_model=ONNXClassificationModel(model_type="nickname",
                                           base_path=root_dir+'/model')
    nickname_model.load()
    
    text=input("검증할 닉네임을 입력해주세요: ")
    nickname=re.sub(r"-._", " ", text)
    print(nickname_model.predict(text))