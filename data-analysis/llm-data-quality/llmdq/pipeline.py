import sys
from typing import List, Tuple
import logging
from random import choice
from llmdq.config import FilterConfig
from llmdq.struct import InstructAnswer
from llmdq.scorer import RewardModelScorer, PerplexityScorer, ToxicityScorer, GibberishScorer, LengthScorer, ScorerPipeline
from llmdq.scorefilter import FilterPipeline
from llmdq.clustering import Dedup, SemanticKmeansClustering, ClusteringPipeline


lg = logging.getLogger(__name__)
lg.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s'))
lg.addHandler(handler)


def llmdq_pipeline(data: List[InstructAnswer], config: FilterConfig, batch_size, device) -> Tuple[List[InstructAnswer], List[InstructAnswer]]:
    lg.info("Scoring has started")
    scorer_pipeline = ScorerPipeline()
    scorer_pipeline.add([
        RewardModelScorer("OpenAssistant/reward-model-deberta-v3-large", batch_size=batch_size, device=device),
        PerplexityScorer("gpt2", batch_size=batch_size, device=device),
        ToxicityScorer("unitary/toxic-bert", batch_size=batch_size, device=device),
        GibberishScorer("madhurjindal/autonlp-Gibberish-Detector-492513457", batch_size=batch_size, device=device),
        LengthScorer()]
    )
    scorer_pipeline.score(data)

    lg.info("Filtering has started")
    filterpipe = FilterPipeline(config)
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
    clusteringpipe.add([
        Dedup(),
        SemanticKmeansClustering("facebook/contriever",
                                 batch_size=batch_size,
                                 device=device)
    ])
    clustered_data = clusteringpipe.run(filtered_dataset)
    lg.info("Pipeline has finished")
    return clustered_data, removed_dataset