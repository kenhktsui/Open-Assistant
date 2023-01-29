from typing import List, Iterable
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm
from llmdq.struct import InstructAnswer
from llmdq.clustering.base import ClusteringBase


lg = logging.getLogger(__name__)


class Dedup(ClusteringBase):
    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        """TODO"""
        return instructanswer_list


class SemanticKmeansClustering(ClusteringBase):
    """
    Implemented from: https://colab.research.google.com/drive/13eGPGqcHcfJQhqTgX-PnZ5C0Fkb8nVLJ?usp=sharing
    """
    def __init__(self, model_id, batch_size=8, device=-1, n_cluster=100, niter=10, sample_rate=0.1):
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._batch_size = batch_size
        self._device = f"cuda:{device}" if device >= 0 else "cpu"
        self._model.to(self._device)
        self._n_cluster = n_cluster
        self._niter = niter
        self._sampling_rate = sample_rate

    def _batching(self, iterable: list) -> Iterable:
        length = len(iterable)
        for ndx in range(0, length, self._batch_size):
            yield iterable[ndx:min(ndx + self._batch_size, length)]

    def _get_embedding(self, instructanswer_list: List[InstructAnswer]) -> np.ndarray:
        embed_list = []
        full_text = [ia.instruct + "\n" + ia.answer for ia in instructanswer_list]
        for batch_text in tqdm(self._batching(full_text), desc=self.__class__.__name__):
            inputs = self._tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to(self._device)
            embed = self._model(**inputs)
            embed_list.append(embed.pooler_output.cpu().detach().numpy())  # use pooled ouput

        embed = np.vstack(embed_list)

        # normalised with l2-norm
        embed_l2 = np.atleast_1d(np.linalg.norm(embed, ord=2, axis=-1))
        embed_l2[embed_l2 == 0] = 1
        return embed / np.expand_dims(embed_l2, axis=-1)

    def _clustering(self, embeddings: np.ndarray) -> np.ndarray:
        kmeans = faiss.Kmeans(embeddings, self._n_cluster, niter=self._niter, gpu=True if self._device >= 0 else False)
        res = kmeans.train(embeddings)
        _, I = kmeans.index.search(res, 1)
        return I.flatten()

    def _sampling(self, instructanswer_list: List[InstructAnswer], member_list: np.ndarray) -> List[InstructAnswer]:
        targeted_sample_size = len(instructanswer_list) * self._sampling_rate
        sampled_index = set()
        for i in range(self._n_cluster):
            cluster_member = np.where(member_list == i)[0]
            if cluster_member.size <= targeted_sample_size:
                sampled_index.update(cluster_member.tolist())
            else:
                sampled_index.update(np.random.choice(cluster_member, size=targeted_sample_size, replace=False).tolist())
        return [ia for i, ia in enumerate(instructanswer_list) if i in sampled_index]

    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        if len(instructanswer_list) <= self._n_cluster:
            lg.info(f"Data size smaller than or equal to {self._n_cluster}, cannot perform clustering")
            return instructanswer_list
        embeddings = self._get_embedding(instructanswer_list)
        print(embeddings.shape)
        member_list = self._clustering(embeddings)
        sampled_instructanswer_list = self._sampling(instructanswer_list, member_list)
        return sampled_instructanswer_list


class ClusteringPipeline(ClusteringBase):
    def __init__(self):
        self._clustering_list = []

    def add(self, clustering_list) -> None:
        self._clustering_list.extend(clustering_list)

    def run(self, instructanswer_list: List[InstructAnswer]) -> List[InstructAnswer]:
        for clust in self._clustering_list:
            instructanswer_list = clust.run(instructanswer_list)
        return instructanswer_list
