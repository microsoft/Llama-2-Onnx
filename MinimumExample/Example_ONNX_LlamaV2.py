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
    embedding_file: str,
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
    data_type_str = llm_session.get_inputs()[0].type
    if data_type_str == "tensor(float16)":
        data_type = np.float16
    elif data_type_str == "tensor(float32)" or data_type_str == "tensor(float)":
        data_type = np.float32
    else:
        raise Exception(f"Unknown data type {data_type_str}")

    # Get the relevant shapes so we can create the inputs
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name == "x":
            x_shape = inputs_meta.shape
        elif inputs_meta.name == "attn_mask":
            attn_mask_shape = inputs_meta.shape
        elif inputs_meta.name == "k_cache":
            k_cache_shape = inputs_meta.shape

    hidden_size = x_shape[2]
    max_seq_len = attn_mask_shape[1]
    n_layers = k_cache_shape[1]
    n_heads = k_cache_shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=tokenizer_path)
    tokens = tokenizer.encode(prompt, bos=True, eos=False)

    # create the embedding layer.
    embedding_layer = torch.nn.Embedding(tokenizer.n_words, hidden_size)
    embedding_layer.load_state_dict(torch.load(embedding_file))
    embedding_layer.eval()

    # Create the embeddings of the initial prompt.
    x = embedding_layer(torch.tensor(tokens)).detach().cpu().numpy()
    x = np.expand_dims(x, axis=0).astype(data_type)

    # Create the attention mask.
    attn_mask = -10000.0 * torch.triu(
        torch.ones(attn_mask_shape), diagonal=1
    ).cpu().detach().numpy().astype(data_type)

    # Create the K and V caches.
    head_dim = int(hidden_size / n_heads)
    k_cache = np.zeros([1, n_layers, max_seq_len, n_heads, head_dim], dtype=data_type)
    v_cache = np.zeros([1, n_layers, max_seq_len, n_heads, head_dim], dtype=data_type)

    # Iteratively generate tokens.
    pos = np.array(0)
    output_tokens = []
    for idx in range(max_gen_len):
        results = llm_session.run(
            None,
            {
                "x": x,
                "attn_mask": attn_mask,
                "k_cache": k_cache[:, :, :pos],
                "v_cache": v_cache[:, :, :pos],
                "pos": pos.astype(np.int64),
            },
        )
        logits, k_out, v_out = results[:3]

        # Decide the next token using your preferred sampling strategy.
        next_token = np.argmax(logits, axis=-1).astype(np.int64)
        output_tokens.extend(next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if next_token == tokenizer.eos_id:
            break

        # Update the cache
        seq_len = x.shape[1]
        k_cache[:, :, pos : pos + seq_len] = k_out
        v_cache[:, :, pos : pos + seq_len] = v_out

        # Update pos and x ready for the next round.
        pos = np.array(int(pos) + seq_len, dtype=np.int64)
        x = embedding_layer(torch.tensor(next_token)).unsqueeze(0)
        x = x.cpu().detach().numpy().astype(data_type)

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
        "--embedding_file",
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
        args.embedding_file,
        args.tokenizer_path,
        args.max_gen_len,
    )

    print(response)
