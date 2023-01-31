import sys
from typing import List, Tuple
import logging
from random import choice
from llmdq.config import Config
from llmdq.struct import InstructAnswer
from llmdq.scorer import ScorerPipeline
from llmdq.scorefilter import FilterPipeline
from llmdq.clustering import ClusteringPipeline
from llmdq.dynamic_import import instantiate_class_from_config


lg = logging.getLogger(__name__)
lg.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s'))
lg.addHandler(handler)


def llmdq_pipeline(data: List[InstructAnswer], config: Config) -> Tuple[List[InstructAnswer], List[InstructAnswer]]:
    lg.info("Scoring has started")

    _obj_map = instantiate_class_from_config(config)

    scorer_pipeline = ScorerPipeline()
    scorer_pipeline.add(_obj_map['scorer'])
    scorer_pipeline.score(data)

    lg.info("Filtering has started")
    filterpipe = FilterPipeline(config.scorefilter)
    filterpipe.process(data)
    filtered_dataset = filterpipe.get_clean_dataset()
    removed_dataset = filterpipe.get_removed_dataset()

    lg.debug("Examples of good data:")
    for _ in range(min(len(filtered_dataset), 5)):
        lg.debug(choice(filtered_dataset))

    lg.debug("Examples of bad data:")
    for _ in range(min(len(removed_dataset), 5)):
        lg.debug(choice(removed_dataset))

    lg.info("Clustering has started")
    clusteringpipe = ClusteringPipeline()
    clusteringpipe.add(_obj_map['clustering'])
    clustered_data = clusteringpipe.run(filtered_dataset)
    lg.info("Pipeline has finished")
    return clustered_data, removed_dataset
