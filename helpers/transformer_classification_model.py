from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import gc
import os
import pandas as pd
import re

class TransformerClassificationModel:
    def __init__(self, model_type:str, base_path:str="./model"):
        self.model_type=model_type

        self.model_path=os.path.join(base_path, f"{model_type}_model")
        self.tokenizer_path=os.path.join(base_path, f"{model_type}_tokenizer")
        self.dataset_path=os.path.join(base_path, 'dataset.csv')

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.max_length=256 if model_type == "comment" else 40

        self.model=None
        self.tokenizer=None

    def load(self):
        try:
            if self.model:
                return
            
            self.generate_label_array()

            self._raise_file_not_fount(self.tokenizer_path)
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_path)
            
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
                del self.model
                self.model=None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer=None
            
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

    def predict(self, text: str):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")
        
        inputs=self.tokenizer(text,
                              return_tensors="pt",
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_length)
        
        inputs={ key: val.to(self.device) for key, val in inputs.items() }

        with torch.no_grad():
            outputs=self.model(**inputs)
            predicted_class_index=torch.argmax(outputs.logits, dim=1).item()
        
        return self.label_array[predicted_class_index]
    
    def predict_batch(self, texts: list[str]) -> list[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")
        

        batch_size = 128

        all_predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
        
            inputs = self.tokenizer(batch_texts,
                                    return_tensors="pt",
                                    truncation=True,
                                    padding='max_length',
                                    max_length=self.max_length)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_class_indexes = torch.argmax(outputs.logits, dim=1).tolist()

            batch_predictions = [self.label_array[idx] for idx in predicted_class_indexes]
            all_predictions.extend(batch_predictions)
        return all_predictions

if __name__ == "__main__":
    print('cuda available: ', torch.cuda.is_available())

    root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nickname_model=TransformerClassificationModel(model_type="nickname",
                                                  base_path=root_dir+'/model')
    nickname_model.load()

    text=input("검증할 닉네임을 입력해주세요: ")
    nickname=re.sub(r"-._", " ", text)
    predicted=nickname_model.predict(text)
    print(predicted, type(nickname_model.predict(text)))

    