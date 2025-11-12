
from base_evaluator import BaseEvaluator
from transformers import AutoModel

class MusicGenEvaluator(BaseEvaluator):
    
    def __init__(self, name):
        self.name = name
        self.model = 