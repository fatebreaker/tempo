# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/dapo")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "open-r1/DAPO-Math-17k-Processed"
    dataset = datasets.load_dataset(data_path, "en")

    split = dataset["train"].train_test_split(test_size=0.03, seed=42)

    # This gives you a dict with 'train' and 'test'
    train_dataset = split['train']
    test_dataset = split['test']
    # import ipdb; ipdb.set_trace()
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    data_source = "DigitalLearningGmbH/MATH-lighteval"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            orig_extra_info = example.pop("extra_info")
            extra_info = orig_extra_info.copy()
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": example["reward_model"]["ground_truth"],
                    },
                },
            }
            question = example['prompt']  + " " + instruction_following
            example['prompt'] = [{"role": "user", "content": question}]
            example["extra_info"] = extra_info
            example['data_source'] = data_source
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    # import ipdb; ipdb.set_trace()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_split.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test_split.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
