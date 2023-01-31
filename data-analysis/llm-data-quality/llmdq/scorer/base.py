from typing import List, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm
from transformers import pipeline
from llmdq.struct import ScorerOutput, InstructAnswer


class ScorerBase(ABC):

    @abstractmethod
    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        pass


class HFPipelineScorerBase(ScorerBase):

    def __init__(self, score_id: str, model_id: str, task: str, batch_size: int, top_k: Optional[int], device: int):
        self._classifier = pipeline(task, model=model_id, batch_size=batch_size, top_k=top_k, device=device)
        self._model_id = self._classifier.model.name_or_path
        self._score_id = score_id

    @abstractmethod
    def input_preprocessing(self, ia: InstructAnswer):
        """Preprocessing InstructAnswer into text for scorer input"""
        pass

    @abstractmethod
    def score_processing(self, output: dict) -> float:
        """Convert classifier output into float"""
        pass

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [self.input_preprocessing(ia) for ia in instructanswer_list]
        score_list = []
        for i in tqdm(self._classifier(full_text), total=len(full_text), desc=self.__class__.__name__):
            score_list.append(ScorerOutput(model_id=self._model_id, score_id=self._score_id,
                                           score=self.score_processing(i)))
        return score_list