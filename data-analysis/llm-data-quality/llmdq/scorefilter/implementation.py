from typing import List
from llmdq.struct import InstructAnswer
from llmdq.config import FilterConfig
from llmdq.scorefilter.base import ScoreFilterBase
from llmdq.config import AbsoluteFilterConfig, ZScoreFilterConfig


class AbsoluteScoreFilter(ScoreFilterBase):
    filter_type = "absolute"

    def __init__(self, config_list: List[AbsoluteFilterConfig]):
        super().__init__()
        self._config_list = config_list

    def is_pass(self, instructanswer: InstructAnswer) -> bool:
        for score in instructanswer.score:
            for config in self._config_list:
                if score.score_id == config.score_id:
                    if config.direction == "ge" and score.score <= config.threshold:
                        return False
                    if config.direction == "le" and score.score >= config.threshold:
                        return False
        return True


class ZScoreFilter(ScoreFilterBase):
    filter_type = "absolute"

    def __init__(self, config_list: List[ZScoreFilterConfig]):
        super().__init__()
        self._config_list = config_list

    def _get_dataset_statistics(self, instructanswer_list: List[InstructAnswer]) -> None:
        self._dataset_stat = {}

        if not instructanswer_list:
            return

        data_len = len(instructanswer_list)

        for config in self._config_list:
            self._dataset_stat[config.score_id] = {}
            running_x1 = 0
            running_x2 = 0
            for ia in instructanswer_list:
                for s in ia.score:
                    if config.score_id == s.score_id:
                        running_x1 += s.score
                        running_x2 += s.score ** 2

            mean = running_x1/data_len
            std = (running_x2/data_len - mean**2) ** 0.5
            self._dataset_stat[config.score_id]["mean"] = mean
            if std == 0:
                raise Exception(f"Dataset statistics {config.score_id} has zero std")
            self._dataset_stat[config.score_id]["std"] = std

    def is_pass(self, instructanswer: InstructAnswer) -> bool:
        for config in self._config_list:
            score_stat = self._dataset_stat[config.score_id]
            for score in instructanswer.score:
                if score.score_id == config.score_id:
                    z_score = (score.score - score_stat["mean"]) / score_stat["std"]
                    if config.direction == "ge" and z_score <= config.threshold:
                        return False
                    if config.direction == "le" and z_score >= config.threshold:
                        return False
        return True


class FilterPipeline(ScoreFilterBase):
    def __init__(self, config: FilterConfig):
        super().__init__()
        self._absfilter = AbsoluteScoreFilter(config.absolute)
        self._zscorefilter = ZScoreFilter(config.relative)

    def _get_dataset_statistics(self, instructanswer_list):
        self._zscorefilter._get_dataset_statistics(instructanswer_list)

    def is_pass(self, instructanswer: InstructAnswer):
        return (
                self._absfilter.is_pass(instructanswer) and
                self._zscorefilter.is_pass(instructanswer)
                )
