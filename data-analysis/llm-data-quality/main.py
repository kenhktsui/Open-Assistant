if __name__ == "__main__":
    import argparse
    import os
    import json
    from llmdq import FilterConfig, InstructAnswer, llmdq_pipeline

    parser = argparse.ArgumentParser("Data quality pipeline")
    parser.add_argument("filter_config")
    parser.add_argument("in_data_path")
    parser.add_argument("out_data_path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=int, default=-1, help="positive means GPU no, -1 means using CPU")
    args = parser.parse_args()

    with open(args.filter_config) as f:
        config = FilterConfig(**json.load(f))

    with open(args.in_data_path) as f:
        data = json.load(f)
        data = [InstructAnswer(**d) for d in data]

    clustered_data, removed_dataset = llmdq_pipeline(data, config, batch_size=args.batch_size, device=args.device)

    base_name = os.path.splitext(args.in_data_path)[0]
    with open(base_name + "_filtered.json", 'w') as f:
        json.dump([i.dict() for i in clustered_data], f)
    with open(base_name + "_removed.json", 'w') as f:
        json.dump([i.dict() for i in removed_dataset], f)
