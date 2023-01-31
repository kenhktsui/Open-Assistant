if __name__ == "__main__":
    import argparse
    import os
    import json
    import yaml
    from llmdq import Config, InstructAnswer, llmdq_pipeline

    parser = argparse.ArgumentParser("Data quality pipeline")
    parser.add_argument("config")
    parser.add_argument("in_data_path")
    parser.add_argument("out_data_path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = Config(**config)

    with open(args.in_data_path) as f:
        data = json.load(f)
        data = [InstructAnswer(**d) for d in data]

    clustered_data, removed_dataset = llmdq_pipeline(data, config)

    base_name = os.path.splitext(args.in_data_path)[0]
    with open(base_name + "_filtered.json", 'w') as f:
        json.dump([i.dict() for i in clustered_data], f)
    with open(base_name + "_removed.json", 'w') as f:
        json.dump([i.dict() for i in removed_dataset], f)
