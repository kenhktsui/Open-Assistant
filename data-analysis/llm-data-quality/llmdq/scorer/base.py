from typing import List
from abc import ABC, abstractmethod
from tqdm import tqdm
from transformers import pipeline
from llmdq.struct import ScorerOutput, InstructAnswer


class ScorerBase(ABC):

    @abstractmethod
    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        pass


class HFPipelineScorerBase(ScorerBase):

    def __init__(self, score_id: str, model_id: str, task: str, batch_size: int, device: int, **kwargs):
        self._model = pipeline(task, model=model_id, batch_size=batch_size, device=device, **kwargs)
        self._model_id = self._model.model.name_or_path
        self._score_id = score_id

    @abstractmethod
    def input_preprocessing(self, ia: InstructAnswer) -> str:
        """Preprocessing InstructAnswer into text for scorer input"""
        pass

    @abstractmethod
    def score_processing(self, output: dict) -> float:
        """Convert classifier output into float to cater for different output in HF model hub"""
        pass

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [self.input_preprocessing(ia) for ia in instructanswer_list]
        score_list = []
        for i in tqdm(self._model(full_text), total=len(full_text), desc=self.__class__.__name__):
            score_list.append(ScorerOutput(model_id=self._model_id, score_id=self._score_id,
                                           score=self.score_processing(i)))
        return score_list
