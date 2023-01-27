from typing import List
from abc import ABC, abstractmethod
from llmdq.struct import InstructAnswer


class ClusteringBase(ABC):
    score_type = "base"

    @abstractmethod
    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        pass
