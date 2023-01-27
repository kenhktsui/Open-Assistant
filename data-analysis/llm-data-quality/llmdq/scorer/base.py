from typing import List
from abc import ABC, abstractmethod
from llmdq.struct import ScorerOutput


class ScorerBase(ABC):
    score_type = "base"

    @abstractmethod
    def score(self, instructanswer_list) -> List[ScorerOutput]:
        pass
