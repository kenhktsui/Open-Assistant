from abc import ABC, abstractmethod
from typing import List
from llmdq.struct import InstructAnswer


class ScoreFilterBase(ABC):
    filter_type = "base"

    def __init__(self):
        self._clean_dataset = []
        self._removed_dataset = []

    @abstractmethod
    def is_pass(self, instructanswer: InstructAnswer) -> None:
        pass

    def _get_dataset_statistics(self, instructanswer_list) -> None:
        pass

    def process(self, instructanswer_list: List[InstructAnswer]) -> None:
        """removal not inplace yet"""
        self._get_dataset_statistics(instructanswer_list)

        for ia in instructanswer_list:
            if self.is_pass(ia):
                self._clean_dataset.append(ia)
            else:
                self._removed_dataset.append(ia)

    def get_clean_dataset(self):
        return self._clean_dataset

    def get_removed_dataset(self):
        """ For inspection"""
        return self._removed_dataset
