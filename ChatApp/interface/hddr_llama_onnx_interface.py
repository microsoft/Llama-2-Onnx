import torch
import onnxruntime
import numpy as np
from sentencepiece import SentencePieceProcessor
from typing import List
import os
import logging
import gc

from .base_interface import BaseLLMInterface

from app_modules.utils import (
    is_stop_word_or_prefix,
    convert_to_markdown,
    shared_state,
)


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


class LlamaOnnxInterface(BaseLLMInterface):
    def __init__(self, onnx_file="", embedding_file="", tokenizer_path=""):
        super().__init__()

        self.onnx_file = onnx_file
        self.embedding_file = embedding_file
        self.tokenizer_path = tokenizer_path

        self.total_count = 0

    def initialize(self):
        # Create the ONNX session

        logging.info(f"Creating ONNX session for [{self.onnx_file}]")
        options = onnxruntime.SessionOptions()
        self.llm_session = onnxruntime.InferenceSession(
            self.onnx_file,
            sess_options=options,
            providers=[
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        # get the data type used by the model
        data_type_str = self.llm_session.get_inputs()[0].type
        if data_type_str == "tensor(float16)":
            self.data_type = np.float16
        elif data_type_str == "tensor(float32)":
            self.data_type = np.float32
        else:
            raise Exception(f"Unknown data type {data_type_str}")

        logging.info(f"Detected Data Type [{self.data_type}]")

        # Get the relevant shapes so we can create the inputs
        for inputs_meta in self.llm_session._inputs_meta:
            if inputs_meta.name == "x":
                x_shape = inputs_meta.shape
            elif inputs_meta.name == "attn_mask":
                attn_mask_shape = inputs_meta.shape
            elif inputs_meta.name == "k_cache":
                k_cache_shape = inputs_meta.shape

        self.hidden_size = x_shape[2]
        self.max_seq_len = attn_mask_shape[1]
        self.n_layers = k_cache_shape[1]
        self.n_heads = k_cache_shape[3]

        # Initialize the tokenizer and produce the initial tokens.
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)

        # create the embedding layer.
        logging.info(
            f"Creating the Embedding Layer. Size [{self.tokenizer.n_words}, {self.hidden_size}]"
        )
        self.embeddingLayer = torch.nn.Embedding(
            self.tokenizer.n_words, self.hidden_size
        )

        # rg hack - dont have the embeddings.pth file - taking it from the original llama model
        d = torch.load(self.embedding_file)
        self.embeddingLayer.load_state_dict(d)
        self.embeddingLayer.eval()

        # Create the attention mask.
        self.attn_mask = -10000.0 * torch.triu(
            torch.ones(attn_mask_shape), diagonal=1
        ).cpu().detach().numpy().astype(self.data_type)

        # Create the K and V caches.
        self.head_dim = int(self.hidden_size / self.n_heads)
        self.k_cache = np.zeros(
            [1, self.n_layers, self.max_seq_len, self.n_heads, self.head_dim],
            dtype=self.data_type,
        )
        self.v_cache = np.zeros(
            [1, self.n_layers, self.max_seq_len, self.n_heads, self.head_dim],
            dtype=self.data_type,
        )

    def shutdown(self):
        pass

    def generate_prompt_with_history(self, text, history, tokenizer, max_length=2048):
        prompt = "[|Human|]Hey there I am a human that would like to have\
a conversation with you.\n[|AI|]Sure, I am happy to answer most questions\
\n[|Human|]Great, I insist that we take turns.\n[|AI|]I agree, we should\
 take turns.\n[|Human|]Great, can we also keep answers short\n[|AI|]Yes, \
short answers are usually best"

        history = ["\n[|Human|]{}\n[|AI|]{}".format(x[0], x[1]) for x in history]
        history.append("\n[|Human|]{}\n[|AI|]".format(text))
        history_text = ""
        flag = False
        for x in history[::-1]:
            # tokens = self.tokenizer.encode(text, bos=True, eos=False)
            if (
                len(
                    self.tokenizer.encode(
                        prompt + history_text + x, bos=True, eos=False
                    )
                )
                <= max_length
            ):
                history_text = x + history_text
                flag = True
            else:
                break
        if flag:
            return prompt + history_text, torch.tensor(
                self.tokenizer.encode(prompt + history_text, bos=True, eos=False)
            ).unsqueeze(0)
        else:
            return None

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None

            # temperature, converting to probabilities and sorting are common to both top-k and top-p
            # convert logits to 32-bit float to avoid numerical issues with np.exp
            logits = logits.astype(np.float32)
            # Scale the logits by the temperature
            logits /= temperature
            # Convert logits to probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))
            # Sort th probabilities and indexes
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            # find the index of interest for each of the methods.
            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                # find the value of the first cumalitive probability that exceeds p
                for index_of_interest, cumulative_prob in enumerate(
                    cumulative_probs[0]
                ):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            # Normalize the probabilities and select the next token
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array(
                [np.random.choice(indices_of_interest[0], p=probs_of_interest[0])]
            )
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    def greedy_search(
        self,
        input_ids,
        model,
        tokenizer,
        stop_words: list,
        max_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 25,
    ):
        generated_tokens = []
        pos = np.array(0)

        x = (
            self.embeddingLayer(torch.tensor(input_ids))
            .detach()
            .cpu()
            .numpy()
            .astype(self.data_type)
        )

        for i in range(max_length):
            results = self.llm_session.run(
                None,
                {
                    "x": x,
                    "attn_mask": self.attn_mask,
                    "k_cache": self.k_cache[:, :, :pos],
                    "v_cache": self.v_cache[:, :, :pos],
                    "pos": pos.astype(np.int64),
                },
            )
            logits, k_out, v_out = results[:3]

            next_token = self.sample_logits(logits, "top_p", top_p, temperature)
            next_token = next_token.reshape(1, -1)

            # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
            if next_token[0] == tokenizer.eos_id:
                del logits
                gc.collect()
                return

            input_ids = torch.cat((input_ids, torch.tensor(next_token)), dim=-1)

            generated_tokens.append(next_token[0].item())
            text = tokenizer.decode(generated_tokens)

            seq_len = x.shape[1]
            self.k_cache[:, :, pos : pos + seq_len] = k_out
            self.v_cache[:, :, pos : pos + seq_len] = v_out
            pos = np.array(int(pos) + seq_len)

            x = (
                self.embeddingLayer(torch.tensor(next_token))
                .unsqueeze(0)
                .reshape([1, 1, self.hidden_size])
                .cpu()
                .detach()
                .numpy()
                .astype(self.data_type)
            )

            yield text

            if any([x in text for x in stop_words]):
                del logits
                gc.collect()
                return

    def predict(
        self,
        text,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        if text == "":
            yield chatbot, history, "Empty context."
            return
        try:
            self.llm_session
        except (ValueError, RuntimeError, TypeError):
            yield [[text, "No Model Found"]], [], "No Model Found"
            return

        inputs = self.generate_prompt_with_history(
            text, history, self.tokenizer, max_length=max_context_length_tokens
        )

        if inputs is None:
            yield chatbot, history, "Input too long."
            return
        else:
            prompt, inputs = inputs

        input_ids = inputs[:, -max_context_length_tokens:]

        # global total_count
        self.total_count += 1
        print(self.total_count)

        self.head_dim = int(self.hidden_size / self.n_heads)
        self.k_cache = np.zeros(
            [1, self.n_layers, self.max_seq_len, self.n_heads, self.head_dim],
            dtype=self.data_type,
        )
        self.v_cache = np.zeros(
            [1, self.n_layers, self.max_seq_len, self.n_heads, self.head_dim],
            dtype=self.data_type,
        )

        x = input_ids

        for x in self.greedy_search(
            input_ids,
            self.llm_session,
            self.tokenizer,
            stop_words=["[|Human|]", "[|AI|]"],
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            if is_stop_word_or_prefix(x, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in x:
                    x = x[: x.index("[|Human|]")].strip()
                if "[|AI|]" in x:
                    x = x[: x.index("[|AI|]")].strip()
                x = x.strip()
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(x)]
                ], history + [[text, x]]
                yield a, b, "Generating..."
            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except Exception as e:
                    print(type(e).__name__, e)
                    pass

        del input_ids
        gc.collect()
        torch.cuda.empty_cache()

        try:
            yield a, b, "Generate: Success"
        except Exception as e:
            print(type(e).__name__, e)
            pass

        return

    def retry(
        self,
        text,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        logging.info("Retry...")
        if len(history) == 0:
            yield chatbot, history, "Empty context"
            return
        chatbot.pop()
        inputs = history.pop()[0]
        for x in self.predict(
            inputs,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ):
            yield x
