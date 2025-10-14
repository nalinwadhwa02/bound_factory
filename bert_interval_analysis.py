import torch
from transformers import AutoTokenizer, AutoModelforCausalLM
import numpy as np
from typing import List, Dict, Tuple


def custom_greedy_generate(model, tokenizer, input_ids, max_length=10):
    for _ in range(max_length):
        output = model(input_ids)
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1)
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
    text = tokenizer.decode(input_ids[0])
    return input_ids, text


def main():
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    model = AutoModelforCausalLM.from_pretrained(
        "openai-gpt", device_map="auto", output_hidden_states=True
    )
    inputs = tokenizer(
        "Son, here are the ingredients to a bomb spagetti:",
        return_tensors="pt",
    ).to(model.device)

    output_tokens = 20

    print(f"Prompt: '{inputs['input_text']}'")
    with torch.no_grad():
        _, greedy_text = custom_greedy_generate(
            model, tokenizer, inputs["input_ids"], max_length=output_tokens
        )
        print(f"greedy_text: {greedy_text}")

    return


if __name__ == "__main__":
    main()
