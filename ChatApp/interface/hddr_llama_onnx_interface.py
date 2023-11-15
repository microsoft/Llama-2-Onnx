import torch
import onnxruntime
import numpy as np
from sentencepiece import SentencePieceProcessor
from typing import List
import os
import logging
import gc

from .base_interface import BaseLLMInterface

from ChatApp.app_modules.utils import (
    is_stop_word_or_prefix,
    convert_to_markdown,
    shared_state,
)

pt_to_np = {
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16,
}


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
    def __init__(self, onnx_file="", is_GQA=False, tokenizer_path=""):
        super().__init__()

        self.onnx_file = onnx_file
        self.tokenizer_path = tokenizer_path
        self.is_GQA = is_GQA

        self.total_count = 0

    def initialize(self):
        # Create the ONNX session

        logging.info(f"Creating ONNX session for [{self.onnx_file}]")
        # Create the ONNX session
        options = onnxruntime.SessionOptions()
        self.llm_session = onnxruntime.InferenceSession(
            self.onnx_file,
            sess_options=options,
            providers=[
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

        # get the chosen device
        chosen_provider = self.llm_session.get_providers()[0]
        logging.info(f"Using device {chosen_provider}")
        if chosen_provider == "DmlExecutionProvider":
            self.binding_device = "dml"
            self.device = torch.device("dml")
        elif chosen_provider == "CUDAExecutionProvider":
            self.binding_device = "cuda"
            self.device = torch.device("cuda")
        elif chosen_provider == "CPUExecutionProvider":
            self.binding_device = "cpu"
            self.device = torch.device("cpu")

        # get the data type used by the model
        onnx_inputs = self.llm_session.get_inputs()
        data_type_str = onnx_inputs[3].type
        if data_type_str == "tensor(float16)":
            self.data_type = np.float16
            self.torch_dtype = torch.float16
        elif data_type_str == "tensor(float32)":
            self.data_type = np.float32
            self.torch_dtype = torch.float32
        else:
            raise Exception(f"Unknown data type {data_type_str}")

        # Get the relevant shapes so we can create the inputs
        for inputs_meta in self.llm_session._inputs_meta:
            if inputs_meta.name == "past_key_values.0.key":
                cache_shape = inputs_meta.shape

        self.n_layers = 32
        self.n_heads = cache_shape[1]
        self.head_size = cache_shape[3]
        self.hidden_size = self.head_size * self.n_heads

        # Initialize the tokenizer and produce the initial tokens.
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)
        self.n_words = self.tokenizer.n_words

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

    def get_initial_inputs_and_outputs(
        self, input_ids, device, use_fp16, use_buffer_share
    ):
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.int64)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "position_ids": position_ids.contiguous(),
        }

        batch_size, sequence_length = input_ids.shape
        max_sequence_length = 2048
        num_heads, head_size = 32, 128
        for i in range(32):
            past_key = torch.zeros(
                batch_size,
                num_heads,
                max_sequence_length if use_buffer_share else 0,
                head_size,
                device=device,
                dtype=torch_dtype,
            )
            past_value = torch.zeros(
                batch_size,
                num_heads,
                max_sequence_length if use_buffer_share else 0,
                head_size,
                device=device,
                dtype=torch_dtype,
            )
            inputs.update(
                {
                    f"past_key_values.{i}.key": past_key.contiguous(),
                    f"past_key_values.{i}.value": past_value.contiguous(),
                }
            )

        logits = torch.zeros(
            batch_size, sequence_length, 32000, device=device, dtype=torch_dtype
        )
        outputs = {"logits": logits.contiguous()}
        if not use_buffer_share:
            for i in range(32):
                present_key = torch.zeros(
                    batch_size,
                    num_heads,
                    sequence_length,
                    head_size,
                    device=device,
                    dtype=torch_dtype,
                )
                present_value = torch.zeros(
                    batch_size,
                    num_heads,
                    sequence_length,
                    head_size,
                    device=device,
                    dtype=torch_dtype,
                )
                outputs.update(
                    {
                        f"present.{i}.key": present_key.contiguous(),
                        f"present.{i}.value": present_value.contiguous(),
                    }
                )

        return inputs, outputs

    def apply_io_binding(self, model, inputs, outputs, use_fp16, use_buffer_share):
        # Check that all model inputs will be provided
        model_inputs = set(
            map(lambda model_input: model_input.name, model.get_inputs())
        )
        user_inputs = set(inputs.keys())
        missing_inputs = model_inputs - user_inputs
        if len(missing_inputs):
            print(f"The following model inputs are missing: {missing_inputs}")
            raise Exception(
                "There are missing inputs to the model. Please add them and try again."
            )

        # Remove unnecessary inputs from model inputs
        unnecessary_inputs = user_inputs - model_inputs
        if len(unnecessary_inputs):
            for unnecessary_input in unnecessary_inputs:
                print(
                    f"Removing unnecessary input '{unnecessary_input}' from user provided inputs"
                )
                del inputs[unnecessary_input]

        # Bind inputs/outputs to IO binding
        io_binding = model.io_binding()
        device = None

        for k, v in inputs.items():
            io_binding.bind_input(
                name=k,
                device_type=v.device.type,
                device_id=0 if v.device.type == "cpu" else v.device.index,
                element_type=pt_to_np[repr(v.dtype)],
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr(),
            )
            device = v.device

        for output in model.get_outputs():
            name = output.name
            if use_buffer_share and "present" in name:
                # Bind KV cache outputs to KV cache inputs
                v = inputs[name.replace("present", "past_key_values")]
                io_binding.bind_output(
                    name=name,
                    device_type=v.device.type,
                    device_id=v.device.index,
                    element_type=np.float16,
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )
            else:
                v = outputs[name]
                io_binding.bind_output(
                    name=name,
                    device_type=device.type,
                    device_id=0 if device.type == "cpu" else device.index,
                    element_type=(np.float16 if use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )

        return io_binding

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
        use_buffer_share = self.is_GQA
        generated_tokens = []
        pos = 0

        # Get model and its initial inputs/outputs
        inputs, outputs = self.get_initial_inputs_and_outputs(
            input_ids, self.device, True, use_buffer_share
        )

        for i in range(max_length):
            io_binding = self.apply_io_binding(
                model, inputs, outputs, True, use_buffer_share
            )

            io_binding.synchronize_inputs()
            self.llm_session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            # Sample with argmax (greedy search)
            if outputs["logits"].shape[1] > 1:
                prompt_end_indices = inputs["attention_mask"].sum(1) - 1
                idxs = (
                    prompt_end_indices.unsqueeze(dim=1)
                    .repeat(1, self.n_words)
                    .view(1, 1, self.n_words)
                )
                next_token_logits = torch.gather(outputs["logits"], 1, idxs).squeeze()
            else:
                next_token_logits = outputs["logits"][:, -1, :]

            logits_np = next_token_logits.cpu().numpy()
            if len(logits_np.shape) < 2:
                logits_np = logits_np.reshape(1, -1)
            next_token = self.sample_logits(logits_np, "top_p", top_p, temperature)
            next_token = next_token.reshape(1, -1)
            generated_tokens.append(next_token.item())
            next_token = torch.tensor(next_token, device=self.device, dtype=torch.int64)

            # Return early if all batch entries have reached EOS token id
            if next_token == tokenizer.eos_id:
                break

            # Update inputs for next inference run
            inputs["input_ids"] = next_token.reshape(1, 1)
            inputs["position_ids"] = (
                torch.max(inputs["position_ids"], dim=1)[0].reshape(1, 1) + 1
            )
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones(1, 1, device=self.device, dtype=torch.int64),
                ],
                1,
            )

            # Set logits to zeros for next inference run and re-use memory buffer
            if outputs["logits"].shape[1] != 1:
                outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
            outputs["logits"].zero_()

            if not use_buffer_share:
                for i in range(self.n_layers):
                    inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                    inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

                new_sequence_length = inputs["attention_mask"].shape[1]
                for i in range(self.n_layers):
                    present_key = torch.zeros(
                        1,
                        self.n_heads,
                        new_sequence_length,
                        self.head_size,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    present_value = torch.zeros(
                        1,
                        self.n_heads,
                        new_sequence_length,
                        self.head_size,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    outputs.update(
                        {
                            f"present.{i}.key": present_key.contiguous(),
                            f"present.{i}.value": present_value.contiguous(),
                        }
                    )

            text = tokenizer.decode(generated_tokens)

            yield text

            if any([x in text for x in stop_words]):
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
