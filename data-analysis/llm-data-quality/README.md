# LLM Data Quality Pipeline (WIP)
## Motivation
To create a data quality framework and pipeline that could combine the best of everyone's code, while remains traceable and easily reproducible.

## Some principles
- extensible to different scorers/ filters/ clustering in whatever framework/ models, so everyone can contribute)
- reproducible and traceable via configuration management (which scorer, filter, clustering config)
- as part of experiment artifact for quicker iteration
- for meta analysis (like does diversity result in better model? does more reward model result in better model?)

## Proposed pipeline:
Config -> `ScorerPipeline` -> `FilterPipeline` (Removal) -> `ClusteringPipeline` -> Human QC/ label -> Some proxy data quality model training -> Good data -> Next Iteration

- Config so far only includes Filter, but that would extend to scorer/ clustering in the future.
- ScorePipeline involves a process of matching instruct+answer to a float, as such it includes reward model, perplexity, safety model, etc.
- FilterPipeline offers two approaches so far, based on absolute threshold, and zscore of scores.
- ClusteringPipeline involves any process involves pairwise comparison, including deduplication, semantic clustering, etc. This step involves sampling and removal of data point.

## How to add your module:
Inherit abstract class and add your implementation in `implementation.py`. Follow the signature to avoid break.

## How to run
```shell
python main.py test_data/test_filter_config.json test_data/test_data.json test_data
```
The filtered data and the removed data are saved for analysis/ quality check/ develop of quality classifier.

## TODO
- A lot of implementations
- performance optimisation (multiprocessing/ offload to GPU)
- Sampling after clustering, ranking
