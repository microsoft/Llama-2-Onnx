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
    options.add_free_dimension_override_by_name("seq_len_increment", 1)
    llm_session = onnxruntime.InferenceSession(
        onnx_file,
        sess_options=options,
        providers=[
            (
                "DmlExecutionProvider",
                {
                    "enable_dynamic_graph_fusion": True,
                    "device_id": 0,
                },
            ),
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
        if inputs_meta.name == "cache.0.key":
            # get the data type of the model.
            if inputs_meta.type == "tensor(float16)":
                data_type = np.float16
            elif inputs_meta.type == "tensor(float32)":
                data_type = np.float32
            else:
                raise Exception(f"Unknown data type {data_type_str}")

            cache_shape = inputs_meta.shape

    n_layers = 32
    n_heads = cache_shape[1]
    head_dim = cache_shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = np.asarray(tokens, dtype=np.int64)
    tokens = np.expand_dims(tokens, axis=0)

    # Create the K and V caches.
    k_caches = [None] * n_layers
    v_caches = [None] * n_layers

    use_cache_branch = np.zeros([1], dtype=np.bool_)

    seq_len = tokens.shape[1]

    tokens_increment = np.array(seq_len, dtype=np.int64).reshape((1, 1))
    position_ids_increment = np.array(0, dtype=np.int64).reshape((1, 1))

    padding = 512

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        # Setup the caches
        if idx == 0 or seq_len % padding == 0:
            padded_seq_len = padding * (seq_len // padding + 1)

            # Create the attention mask, which contains 1's for values that should stay intact, and 0's for values
            # that should get added to -10000
            attn_mask = np.pad(
                np.ones((1, seq_len)), ((0, 0), (padded_seq_len - seq_len, 0))
            ).astype(np.int32)

            for layer_idx in range(n_layers):
                if idx == 0:
                    k_caches[layer_idx] = np.zeros(
                        (1, n_heads, padded_seq_len, head_dim), dtype=data_type
                    )
                    v_caches[layer_idx] = np.zeros(
                        (1, n_heads, padded_seq_len, head_dim), dtype=data_type
                    )
                else:
                    k_caches[layer_idx] = np.pad(
                        k_caches[layer_idx].numpy(),
                        ((0, 0), (0, 0), (padding, 0), (0, 0)),
                    )
                    v_caches[layer_idx] = np.pad(
                        v_caches[layer_idx].numpy(),
                        ((0, 0), (0, 0), (padding, 0), (0, 0)),
                    )

        if idx == 0:
            position_ids = np.arange(seq_len, dtype=np.int64).reshape((1, seq_len))
        else:
            position_ids_increment = np.array(seq_len, dtype=np.int64).reshape((1, 1))

        input_tensors = {
            "tokens": tokens,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "tokens_increment": tokens_increment,
            "position_ids_increment": position_ids_increment,
            "use_cache_branch": use_cache_branch,
        }
        for i in range(n_layers):
            input_tensors[f"cache.{i}.key"] = k_caches[i]
            input_tensors[f"cache.{i}.value"] = v_caches[i]

        results = llm_session.run(
            None,
            input_tensors,
        )
        logits, attn_mask_out = results[:2]
        for i in range(n_layers):
            k_caches[i] = results[2 + 2 * i]
            v_caches[i] = results[3 + 2 * i]

        # Decide the next token using your preferred sampling strategy.
        next_token = np.argmax(logits, axis=-1).astype(np.int64)
        output_tokens.extend(next_token)

        if output_tokens[-1] == tokenizer.eos_id:
            break

        if idx == 0:
            use_cache_branch = np.ones([1], dtype=np.bool_)

        attn_mask_out, attn_mask = attn_mask, attn_mask_out

        tokens_increment = np.expand_dims(next_token, axis=0)
        seq_len += 1

    output_str = tokenizer.decode(torch.tensor(output_tokens).tolist())

    return output_str


if __name__ == "__main__":
    print(
        "Disclaimer: This simple example will not be performant, best performance is achieved with iobinding. \
Please see the ChatApp example for a more complete example. This example is meant to show how the \
model works with the minimum amount of code."
    )

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
