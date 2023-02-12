from typing import Iterable, List
from tqdm import tqdm
from evaluate import load
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
import numpy as np
from llmdq.scorer.base import ScorerBase, HFPipelineScorerBase


class RewardModelScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=1, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

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

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=f"{self.__class__.__name__}_preprocessing")
        output = self._perplexity.compute(data=instructanswer_dataset['text'], model_id=self._model_id,
                        batch_size=self._batch_size, max_length=self._max_length)
        instructanswer_dataset = instructanswer_dataset.remove_columns("text")
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output['perplexities'])
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output['perplexities']))
        return instructanswer_dataset


class ToxicityScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score_processing(self, output: dict) -> float:
        return max([s["score"] for s in output])


class GibberishScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="text-classification", batch_size=8, max_length=1024, top_k=None, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length, top_k=top_k)

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["instruct"] + '\n' + ia["answer"]}

    def score_processing(self, output: dict) -> float:
        return 1.0 - [l for l in output if l['label'] == 'clean'][0]['score']


class ContradictionScorer(HFPipelineScorerBase):
    def __init__(self, score_id, model_id, task="zero-shot-classification", batch_size=8, max_length=1024, device=-1):
        super().__init__(score_id, model_id, task, batch_size, device, max_length)
        self._label_name = ["entailment", "neutral", "contradiction"]

    def input_preprocessing(self, ia: dict) -> dict:
        return {"text": ia["answer"]}

    def score_processing(self, output: dict) -> float:
        idx = output['labels'].index('contradiction')
        return output['scores'][idx]

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=f"{self.__class__.__name__}_preprocessing")
        output = []
        for out in tqdm(self._model(KeyDataset(instructanswer_dataset, "text"), self._label_name, multi_label=False,
                                    batch_size=self._batch_size, max_length=self._max_length),
                        total=len(instructanswer_dataset), desc=self.__class__.__name__):
            output.append(self.score_processing(out))
        instructanswer_dataset = instructanswer_dataset.remove_columns("text")
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output)
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output))
        return instructanswer_dataset


class ReplacedTokenScorer(ScorerBase):
    def __init__(self, score_id, model_id, max_length=512, batch_size=8, device=-1):
        self._score_id = score_id
        self._model_id = model_id
        self._discriminator = ElectraForPreTraining.from_pretrained(model_id)
        self._tokenizer = ElectraTokenizerFast.from_pretrained(model_id)
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._device_pt = f"cuda:{self._device}" if self._device >= 0 else "cpu"
        self._discriminator.to(self._device_pt)

    def _run_discriminator(self, text_batch: List[str]):
        prob_list = []
        inputs = self._tokenizer(text_batch, return_tensors="pt", truncation=True, padding=True,
                                 max_length=self._max_length).to(self._device_pt)
        discriminator_outputs_real = self._discriminator(**inputs)
        for prob in torch.sigmoid(discriminator_outputs_real.logits).cpu().detach().tolist():
            prob_list.append(np.mean(prob))
        return prob_list

    def _batching(self, iterable: list) -> Iterable:
        length = len(iterable)
        for ndx in range(0, length, self._batch_size):
            yield iterable[ndx:min(ndx + self._batch_size, length)]

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        output = []
        for d in tqdm(self._batching(instructanswer_dataset),
                      desc=self.__class__.__name__,
                      total=len(instructanswer_dataset)//self._batch_size+1):
            output.extend(self._run_discriminator(d["answer"]))
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_score", output)
        instructanswer_dataset = instructanswer_dataset.add_column(f"{self._score_id}_model_id", [self._model_id] * len(output))
        return instructanswer_dataset


class LengthScorer(ScorerBase):
    def __init__(self, score_id: str):
        self._score_id = score_id

    def input_preprocessing(self, ia: dict) -> dict:
        return {
            f"{self._score_id}_score": len(ia["answer"]),
            f"{self._score_id}_model_id": "rule"
        }

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        instructanswer_dataset = instructanswer_dataset.map(self.input_preprocessing,
                                                            desc=self.__class__.__name__)
        return instructanswer_dataset


class ScorerPipeline:
    def __init__(self):
        self._scorer_list = []

    def add(self, scorer_list) -> None:
        self._scorer_list.extend(scorer_list)

    def score(self, instructanswer_dataset: Dataset) -> Dataset:
        for scorer in self._scorer_list:
            instructanswer_dataset = scorer.score(instructanswer_dataset)
        return instructanswer_dataset
