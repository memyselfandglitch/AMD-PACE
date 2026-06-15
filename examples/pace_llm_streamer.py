# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

import time

import torch
from transformers import AutoTokenizer, TextStreamer

import pace  # noqa: F401
from pace.llm import LLMModel, SamplingConfig, KVCacheType


def generate_and_time(model, input_encoded, gen_kwargs, text_streamer, pace_model=True):
    """
    Generate output from the model and time the operation.
    """
    start = time.time()
    if pace_model:
        output = model.generate(input_encoded, gen_kwargs, text_streamer)
    else:
        output = model.generate(
            input_encoded.input_ids,
            attention_mask=input_encoded.attention_mask,
            streamer=text_streamer,
            **gen_kwargs,
        )
    print("Time taken: ", time.time() - start)
    return output


def encode_inputs(tokenizer, inputs):
    """
    Encode inputs using the tokenizer.
    """
    return tokenizer(inputs, return_tensors="pt", padding="longest")


def run_pace_llm():
    """
    Run the PACE LLM model.
    """

    model_name = "facebook/opt-6.7b"
    torch_dtype = torch.bfloat16
    inputs_str = [
        "This is an example of using streamer for generating text with llms.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    input_encoded = encode_inputs(tokenizer, inputs_str)

    gen_kwargs = {
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "random_seed": 123,
    }

    kv_cache_type = KVCacheType.BMC
    pace_model = LLMModel(model_name, dtype=torch_dtype, kv_cache_type=kv_cache_type)
    sampling_config = SamplingConfig(**gen_kwargs)
    text_streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    print("\nRunning PACE LLM model")
    print("Model Name: ", model_name)
    print("Input: ", inputs_str)
    print("Encoded Input: ", input_encoded.input_ids)
    print("Attention Mask: ", input_encoded.attention_mask)

    generate_and_time(pace_model, input_encoded, sampling_config, text_streamer)


if __name__ == "__main__":
    run_pace_llm()
