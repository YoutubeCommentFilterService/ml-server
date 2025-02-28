import os
from transformers import AutoTokenizer

class Tokenizer:
    _instance = None

    def __new__(cls, base_path:str = 'model'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            root_path = os.path.expanduser('~/youtube-comment-ml-server')
            cls._instance._tokenizer_path = os.path.join(root_path, base_path, 'tokenizer')
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(cls._instance._tokenizer_path)
        return cls._instance

    def reload_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        return self.get_tokenizer()

    def get_tokenizer(self):
        return self.tokenizer
    
if __name__ == "__main__":
    tokenizer = Tokenizer().get_tokenizer()
    while True:
        text = input("토큰 검증 텍스트 입력(exit): ")
        if text == 'exit':
            break
        print(tokenizer.tokenize(text))