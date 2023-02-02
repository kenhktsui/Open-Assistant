from typing import List, Dict
from evaluate import load
from datasets import Dataset
from llmdq.scorer.base import ScorerBase, HFPipelineScorerBase


class RewardModelScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=1, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, instruct: str, answer: str) -> str:
        return instruct + '\n' + answer

    def score_processing(self, output: dict) -> float:
        return output[0]['score']


class PerplexityScorer(ScorerBase):
    def __init__(self, score_id, model_id, max_length=1024, batch_size=8, device=-1):
        self._perplexity = load("perplexity", module_type="measurement", device=device)
        self._model_id = model_id
        self._batch_size = batch_size
        self._max_length = max_length
        self._score_id = score_id

        # preload the model
        self._perplexity.compute(data=["prewarm model"], model_id=self._model_id)

    def _batch_predict(self, ia_list: Dict[str, List]) -> Dict[str, List]:
        text_input = [instruct + '\n' + answer
                      for instruct, answer in zip(ia_list["instruct"], ia_list["answer"])]
        output = self._perplexity.compute(data=text_input, model_id=self._model_id,
                                          batch_size=self._batch_size, max_length=self._max_length)
        return {
            f"{self._score_id}_score": output['perplexities'],
            f"{self._score_id}_model_id": [self._model_id] * len(output['perplexities'])
        }


class ToxicityScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, instruct: str, answer: str) -> str:
        return instruct + '\n' + answer

    def score_processing(self, output: dict) -> float:
        return max([s["score"] for s in output])


class GibberishScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, instruct: str, answer: str) -> str:
        return instruct + '\n' + answer

    def score_processing(self, output: dict) -> float:
        return 1.0 - [l for l in output if l['label'] == 'clean'][0]['score']


class ContradictionScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="zero-shot-classification", batch_size=8, max_length=1024, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length)
        self._label_name = ["entailment", "neutral", "contradiction"]

    def input_preprocessing(self, instruct: str, answer: str):
        return answer

    def score_processing(self, output: dict) -> float:
        idx = output['labels'].index('contradiction')
        return output['scores'][idx]

    def _batch_predict(self, ia_list: Dict[str, List]) -> Dict[str, List]:
        text_input = [self.input_preprocessing(instruct, answer)
                      for instruct, answer in zip(ia_list["instruct"], ia_list["answer"])]
        output = self._model(text_input, self._label_name, multi_label=False, max_length=self._max_length, truncation=True)
        return {
            f"{self._score_id}_score": list(map(self.score_processing, output)),
            f"{self._score_id}_model_id": [self._model_id] * len(output)
        }


class LengthScorer(ScorerBase):
    def __init__(self, score_id: str):
        self._score_id = score_id

    def _batch_predict(self, ia_list: Dict[str, List]) -> Dict[str, List]:
        return {
            f"{self._score_id}_score": [len(answer) for answer in ia_list["answer"]],
            f"{self._score_id}_model_id": ["rule"] * len(ia_list["answer"])
        }


class ScorerPipeline:
    def __init__(self):
        self._scorer_list = []

    def add(self, scorer_list) -> None:
        self._scorer_list.extend(scorer_list)

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        for scorer in self._scorer_list:
            instructanswer_dataset = scorer.score(instructanswer_dataset)
        return instructanswer_dataset
