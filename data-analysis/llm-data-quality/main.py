import sys
import os
import json
import logging
from random import choice
from llmdq.config import FilterConfig
from llmdq.struct import InstructAnswer
from llmdq.scorer import RewardModelScorer, PerplexityScorer, ToxicityScorer, GibberishScorer, LengthScorer, ScorerPipeline
from llmdq.scorefilter import FilterPipeline
from llmdq.clustering import Dedup, SemanticClustering, ClusteringPipeline


lg = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s'))
lg.addHandler(handler)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Data quality pipeline")
    parser.add_argument("filter_config")
    parser.add_argument("in_data_path")
    parser.add_argument("out_data_path")
    args = parser.parse_args()

    with open(args.filter_config) as f:
        config = FilterConfig(**json.load(f))

    with open(args.in_data_path) as f:
        data = json.load(f)
        data = [InstructAnswer(**d) for d in data]

    lg.info("Scoring has started")
    scorer = ScorerPipeline()
    scorer_pipeline = ScorerPipeline()
    scorer_pipeline.add([
        RewardModelScorer("OpenAssistant/reward-model-deberta-v3-large"),
        PerplexityScorer("gpt2"),
        ToxicityScorer(),
        GibberishScorer(),
        LengthScorer()]
    )
    scorer_pipeline.score(data)
    lg.info("Filtering has started")
    filterpipe = FilterPipeline(config)
    filterpipe.process(data)
    filtered_dataset = filterpipe.get_clean_dataset()
    removed_dataset = filterpipe.get_removed_dataset()

    lg.debug("Examples of good data:")
    for _ in range(5):
        lg.debug(choice(filtered_dataset))

    lg.debug("Examples of bad data:")
    for _ in range(5):
        lg.debug(choice(removed_dataset))

    lg.info("Clustering has started")
    clusteringpipe = ClusteringPipeline()
    clusteringpipe.add([
        Dedup(),
        SemanticClustering()
    ])
    clustered_data = clusteringpipe.run(filtered_dataset)

    base_name = os.path.splitext(args.in_data_path)[0]
    with open(base_name + "_filtered.json", 'w') as f:
        json.dump([i.dict() for i in clustered_data], f)
    with open(base_name + "_removed.json", 'w') as f:
        json.dump([i.dict() for i in removed_dataset], f)