import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from verl.utils.reward_score.math import compute_score

def main(model_path, dataset_name, subset_name=None, split=None, max_new_tokens=128, device="cuda"):
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # Load dataset
    dataset = load_dataset(dataset_name)['train']

    n_total = 0
    n_correct = 0
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    for item in tqdm(dataset, desc="Evaluating"):

        if "AIME-2024" in dataset_name:
            question = item['extra_info']['raw_problem']
            answer = item['reward_model']['ground_truth']

        prompt = [
           {
               "role": "user",
               "content": question + ' ' + instruction_following
           }
        ]

        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        # Exact match
        result = compute_score(response, answer)
        if result > 0:
            n_correct += 1
        n_total += 1

        print(f"\nAccuracy: {n_correct}/{n_total} = {n_correct/n_total:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(**vars(args))
