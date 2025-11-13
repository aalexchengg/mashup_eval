from abc import ABC, abstractmethod
from typing import List

class BaseEvaluator:
    def __init__(self, name):
        self.name = name
    

    def evaluate(samples: List[str], out : str):
        pass

