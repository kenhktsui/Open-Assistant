from typing import List
from llmdq.struct import InstructAnswer, ScorerOutput
from transformers import pipeline
from evaluate import load
from llmdq.scorer.base import ScorerBase


class RewardModelScorer(ScorerBase):
    score_id = "reward"

    def __init__(self, model_id, trun_len=1024):
        self._classifier = pipeline("text-classification", model=model_id)
        self._model_id = model_id
        self._trun_len = trun_len

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        full_text = [t[:self._trun_len] for t in full_text]
        score_list = self._classifier(full_text)
        return [
            ScorerOutput(model_id=self._model_id, score_id=self.score_id, score=i['score']) for i in score_list
            ]


class PerplexityScorer(ScorerBase):
    score_id = "perplexity"

    def __init__(self, model_id, trun_len=1024):
        self._perplexity = load("perplexity", module_type="measurement")
        self._model_id = model_id
        self._trun_len = trun_len

        # preload the model
        self._perplexity.compute(data=["prewarm model"], model_id=self._model_id)

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        full_text = [t[:self._trun_len] for t in full_text]
        score_list = self._perplexity.compute(data=full_text, model_id=self._model_id)
        return [
            ScorerOutput(model_id=self._model_id, score_id=self.score_id, score=i) for i in score_list['perplexities']
            ]


class ToxicityScorer(ScorerBase):
    score_id = 'toxicity'

    def __init__(self, model_id="unitary/toxic-bert", trun_len=1024):
        self._classifier = pipeline("text-classification", model=model_id, top_k=None)
        self._model_id = model_id
        self._trun_len = trun_len

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        full_text = [t[:self._trun_len] for t in full_text]
        score_list = self._classifier(full_text)
        return [
            ScorerOutput(model_id=self._model_id, score_id=self.score_id,
                         score=max([s["score"] for s in i])) for i in score_list
            ]


class GibberishScorer(ScorerBase):
    score_id = 'gibberish'

    def __init__(self, model_id="madhurjindal/autonlp-Gibberish-Detector-492513457", trun_len=1024):
        self._classifier = pipeline("text-classification", model=model_id, top_k=None)
        self._model_id = model_id
        self._trun_len = trun_len

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        full_text = [t[:self._trun_len] for t in full_text]
        score_list = self._classifier(full_text)
        return [
            ScorerOutput(model_id=self._model_id, score_id=self.score_id,
                         score=1.0 - [l for l in i if l['label'] == 'clean'][0]['score']) for i in score_list
            ]


class LengthScorer(ScorerBase):
    score_id = "length"

    def score(self, instructanswer_list: List[InstructAnswer]) -> List[ScorerOutput]:
        full_text = [ia.instruct + '\n' + ia.answer for ia in instructanswer_list]
        score_list = [len(text) for text in full_text]
        return [
            ScorerOutput(model_id="rule", score_id=self.score_id, score=i) for i in score_list
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
