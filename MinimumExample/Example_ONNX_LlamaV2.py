# This program will run the ONNX version of the LlamaV2 model.
import torch
import onnxruntime
import numpy as np
from sentencepiece import SentencePieceProcessor
from typing import List
import os
import argparse


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


def run_onnx_llamav2(
    prompt: str,
    onnx_file: str,
    tokenizer_path: str,
    max_gen_len: int = 256,
) -> str:
    # Create the ONNX session
    options = onnxruntime.SessionOptions()
    llm_session = onnxruntime.InferenceSession(
        onnx_file,
        sess_options=options,
        providers=[
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # get the data type used by the model
    onnx_inputs = llm_session.get_inputs()
    data_type_str = onnx_inputs[3].type
    if data_type_str == "tensor(float16)":
        data_type = np.float16
    elif data_type_str == "tensor(float32)":
        data_type = np.float32
    else:
        raise Exception(f"Unknown data type {data_type_str}")

    # Get the relevant shapes so we can create the inputs
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name == "attention_mask":
            attn_mask_shape = inputs_meta.shape
        elif inputs_meta.name == "past_key_values.0.key":
            cache_shape = inputs_meta.shape

    n_layers = 32
    n_heads = cache_shape[1]
    head_size = cache_shape[3]
    hidden_size = head_size * n_heads

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = np.array(tokens, dtype=np.int64).reshape((1, -1))

    # Create the K and V caches.
    head_dim = int(hidden_size / n_heads)
    kv_cache = [np.zeros((1, n_heads, 0, head_dim), dtype=data_type)] * 2 * n_layers

    # Iteratively generate tokens.
    pos = np.array(0)
    output_tokens = []
    for idx in range(max_gen_len):
        # position_ids is a tensor counting from pos to pos + seq_len
        position_ids = np.arange(pos, pos + tokens.shape[1], dtype=np.int64).reshape(
            (1, -1)
        )
        # Create the attention mask.
        attn_mask = np.ones((1, tokens.shape[1]), dtype=np.int64)
        inputs = {
            "input_ids": tokens,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
        }
        for i in range(0, n_layers):
            inputs[f"past_key_values.{i}.key"] = kv_cache[2 * i]
            inputs[f"past_key_values.{i}.value"] = kv_cache[2 * i + 1]
        results = llm_session.run(
            None,
            inputs,
        )
        logits = results[0]

        for i in range(0, n_layers * 2):
            kv_cache[i] = results[i + 1]

        # Decide the next token using your preferred sampling strategy.
        next_token = np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)
        output_tokens.extend(next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if next_token == tokenizer.eos_id:
            break

        # Update the cache
        seq_len = tokens.shape[1]

        # Update pos and x ready for the next round.
        pos = np.array(int(pos) + seq_len, dtype=np.int64)
        tokens = next_token.reshape((1, -1))

    output_str = tokenizer.decode(torch.tensor(output_tokens).tolist())

    return output_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--onnx_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
    )
    parser.add_argument("--max_gen_len", type=int, default=256)
    args = parser.parse_args()
    response = run_onnx_llamav2(
        args.prompt,
        args.onnx_file,
        args.tokenizer_path,
        args.max_gen_len,
    )

    print(response)
