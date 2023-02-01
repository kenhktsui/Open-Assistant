from typing import List
from evaluate import load
from tqdm import tqdm
from llmdq.struct import InstructAnswer, ScorerOutput
from llmdq.scorer.base import ScorerBase, HFPipelineScorerBase


class RewardModelScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=1, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: InstructAnswer):
        return ia.instruct + '\n' + ia.answer

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

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        full_text = [t for t in full_text]
        score_list = self._perplexity.compute(data=full_text, model_id=self._model_id,
                                              batch_size=self._batch_size, max_length=self._max_length)
        return [
            ScorerOutput(model_id=self._model_id, score_id=self._score_id, score=i) for i in score_list['perplexities']
            ]


class ToxicityScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: InstructAnswer):
        return ia.instruct + '\n' + ia.answer

    def score_processing(self, output: dict) -> float:
        return max([s["score"] for s in output])


class GibberishScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: InstructAnswer):
        return ia.instruct + '\n' + ia.answer

    def score_processing(self, output: dict) -> float:
        return 1.0 - [l for l in output if l['label'] == 'clean'][0]['score']


class ContradictionScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="zero-shot-classification", batch_size=8, max_length=1024, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length)
        self._label_name = ["entailment", "neutral", "contradiction"]

    def input_preprocessing(self, ia: InstructAnswer):
        return ia.answer

    def score_processing(self, output: dict) -> float:
        idx = output['labels'].index('contradiction')
        return output['scores'][idx]

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [self.input_preprocessing(ia) for ia in instructanswer_list]
        score_list = []
        for i in tqdm(self._model(full_text, self._label_name, multi_label=False, max_length=self._max_length),
                      total=len(full_text), desc=self.__class__.__name__):
            score_list.append(ScorerOutput(model_id=self._model_id, score_id=self._score_id,
                                           score=self.score_processing(i)))
        return score_list


class LengthScorer(ScorerBase):
    def __init__(self, score_id: str):
        self._score_id = score_id

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        score_list = [len(text) for text in full_text]
        return [
            ScorerOutput(model_id="rule", score_id=self._score_id, score=i) for i in score_list
            ]


class ScorerPipeline:
    def __init__(self):
        self._scorer_list = []

    def add(self, scorer_list) -> None:
        self._scorer_list.extend(scorer_list)

    def score(self, instructanswer_list: List[InstructAnswer]) -> None:
        results = []
        for scorer in self._scorer_list:
            results.append(scorer.score(instructanswer_list))

        results = list(map(list, zip(*results)))
        for ia, res in zip(instructanswer_list, results):
            ia.score = res
