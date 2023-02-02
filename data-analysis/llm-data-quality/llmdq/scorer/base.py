from typing import List, Dict
from abc import ABC, abstractmethod
from transformers import pipeline
from datasets import Dataset


class ScorerBase(ABC):
    @abstractmethod
    def _batch_predict(self, ia_list: Dict[str, List]) -> Dict[str, List]:
        """
        Input:
        {
            "instruct": ["Hi!", "How are you?"],
            "answer": ["Yo!", "Hey, how are you?"]
        }
        Output:
        {
            "SCORE_ID_score": [0.9, 0.8]
            "SCORE_ID_model_id": ["model1", "model1"]
        }
        """
        pass

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self._batch_predict, batched=True,
                                                            desc=self.__class__.__name__)
        return instructanswer_dataset


class HFPipelineScorerBase(ScorerBase):
    def __init__(self, score_id: str, model_id: str, task: str, batch_size: int, device: int, max_length: int, **kwargs):
        self._model = pipeline(task, model=model_id, device=device, **kwargs)
        self._model_id = self._model.model.name_or_path
        self._score_id = score_id
        self._batch_size = batch_size
        self._max_length = max_length

    @abstractmethod
    def input_preprocessing(self, instruct: str, answer: str) -> str:
        """Preprocessing InstructAnswer into text for scorer input"""
        pass

    @abstractmethod
    def score_processing(self, output: dict) -> float:
        """Convert classifier output into float to cater for different output in HF model hub"""
        pass

    def _batch_predict(self, ia_list: Dict[str, List]) -> Dict[str, List]:
        text_input = [self.input_preprocessing(instruct, answer)
                      for instruct, answer in zip(ia_list["instruct"], ia_list["answer"])]
        output = self._model(text_input, max_length=self._max_length, truncation=True)
        return {
            f"{self._score_id}_score": list(map(self.score_processing, output)),
            f"{self._score_id}_model_id": [self._model_id] * len(output)
        }

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self._batch_predict, batched=True, batch_size=self._batch_size,
                                                            desc=self.__class__.__name__)
        return instructanswer_dataset
