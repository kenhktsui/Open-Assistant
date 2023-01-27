from typing import List
from llmdq.struct import InstructAnswer
from llmdq.clustering.base import ClusteringBase


class Dedup(ClusteringBase):
    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        """TODO"""
        return instructanswer_list


class SemanticClustering(ClusteringBase):
    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        """TODO"""
        return instructanswer_list


class ClusteringPipeline(ClusteringBase):
    def __init__(self):
        self._clustering_list = []

    def add(self, clustering_list) -> None:
        self._clustering_list.extend(clustering_list)

    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        for clust in self._clustering_list:
            instructanswer_list = clust.run(instructanswer_list)
        return instructanswer_list
