from transformers import BertForSequenceClassification, BertTokenizer
import torch

import os
import pandas as pd

class TextClassificationModel:
    def __init__(self, root_dir: str, model_type: str):
        print("hello!")
        self.__ROOT_DIR = root_dir
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "model", f"{model_type}_model")
        self.__model_type = model_type
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__generate_label_array()
        self.__load_model()
        self.__max_length = 120 if model_type == "comment" else 40

    def __load_model(self):
        self.__model = BertForSequenceClassification.from_pretrained(self.__MODEL_DIR)
        self.__tokenizer = BertTokenizer.from_pretrained(self.__MODEL_DIR)

        self.__model.to(self.__device)
        self.__model.eval()

    def eval(self, text: str):
        inputs = self.__tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.__max_length)
        inputs = { key: val.to(self.__device) for key, val in inputs.items() }

        with torch.no_grad():
            outputs = self.__model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        return self.__label_array[predicted_class]
    
    def reload(self):
        del self.__model
        del self.__tokenizer
        self.__generate_label_array()
        self.__load_model()

        print("reload_finished")

    def __generate_label_array(self):
        data_path = os.path.join(self.__ROOT_DIR, "model", "dataset.csv")

        data = pd.read_csv(data_path, usecols=[f"{self.__model_type}_class"])
        self.__label_array = data[f"{self.__model_type}_class"].dropna().unique()
        print(self.__label_array)

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nickname_model = TextClassificationModel(root_dir, "nickname")

    text = input("검증할 닉네임을 입력해주세요: ")
    print(nickname_model.eval(text))

    # nickname_model.reload()

    