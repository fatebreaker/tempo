# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from thefuzz import fuzz

def get_choice(parsed_answer: str, options: dict):
    pred_text = parsed_answer.strip()
    sentences = pred_text.split(".")
    answer = ""
    for sent in sentences:
        if "answer" in sent.lower():
            answer = sent
    if len(answer) == 0:
        answer = parsed_answer
    highest_score = -1
    highest_option = None
    for char, opt in options.items():
        option = char + ": " + opt
        score = max(fuzz.ratio(answer, option), fuzz.ratio(answer, char))
        score = max(score, fuzz.ratio(answer, opt))
        if score > highest_score:
            highest_score = score
            highest_option = char
    return highest_option

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    options = extra_info['options']
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        parsed_answer = match.group(1).strip()
        pred_answer = get_choice(parsed_answer, options)
        if ground_truth == pred_answer:
            return 1.0
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    match = re.match(pattern, solution_str, re.DOTALL)
    if match:
        return 0.0
    return 0.0