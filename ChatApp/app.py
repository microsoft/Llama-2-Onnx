# -*- coding:utf-8 -*-
import os
import logging
import gradio as gr
import gc
from interface.hddr_llama_onnx_interface import LlamaOnnxInterface
from interface.empty_stub_interface import EmptyStubInterface
from ChatApp.app_modules.utils import (
    reset_textbox,
    transfer_input,
    reset_state,
    delete_last_conversation,
    cancel_outputing,
)
from ChatApp.app_modules.presets import (
    small_and_beautiful_theme,
    title,
    description_top,
    description,
)
from ChatApp.app_modules.overwrites import postprocess

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

# we can filter this dictionary at the start according to the actual available files on disk
empty_stub_model_name = "_Empty Stub_"

top_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

tokenizer_path = os.path.join(top_directory, "tokenizer.model")

available_models = {
    "LLaMA 7B HF Float16": {
        "onnx_file": os.path.join(
            top_directory, "7B_HF_CUDA_OPT_float16", "ONNX", "7B_HF_CUDA_OPT_float16.onnx"
        ),
        "tokenizer_path": tokenizer_path,
        "is_GQA": False,
    },
    "LLaMA 7B GQA HF Float16": {
        "onnx_file": os.path.join(
            top_directory, "7B_HF_CUDA_OPT_float16", "ONNX_GQA", "7B_HF_CUDA_OPT_GQA_float16.onnx"
        ),
        "tokenizer_path": tokenizer_path,
        "is_GQA": True,
    },
    "LLaMA 7B HF Float32": {
        "onnx_file": os.path.join(
            top_directory, "7B_HF_CUDA_OPT_float32", "ONNX", "7B_HF_CUDA_OPT_float32.onnx"
        ),
        "tokenizer_path": tokenizer_path,
        "is_GQA": False,
    },
    "LLaMA 7B Chat HF Float16": {
        "onnx_file": os.path.join(
            top_directory, "7B_CHAT_HF_CUDA_OPT_float16", "ONNX", "7B_CHAT_HF_CUDA_OPT_float16.onnx"
        ),
        "tokenizer_path": tokenizer_path,
        "is_GQA": False,
    },
    "LLaMA 7B Chat GQA HF Float16": {
        "onnx_file": os.path.join(
            top_directory, "7B_CHAT_HF_CUDA_OPT_float16", "ONNX_GQA", "7B_CHAT_HF_CUDA_OPT_GQA_float16.onnx"
        ),
        "tokenizer_path": tokenizer_path,
        "is_GQA": True,
    },
    "LLaMA 7B Chat HF Float32": {
        "onnx_file": os.path.join(
            top_directory, "7B_CHAT_HF_CUDA_OPT_float32", "ONNX", "7B_CHAT_HF_CUDA_OPT_float32.onnx"
        ),
        "tokenizer_path": tokenizer_path,
        "is_GQA": False,
    },
}


interface = EmptyStubInterface()
interface.initialize()

# interface = None

gr.Chatbot.postprocess = postprocess

with open("ChatApp/assets/custom.css", "r", encoding="utf-8") as f:
    custom_css = f.read()


def change_model_listener(new_model_name):
    if new_model_name is None:
        new_model_name = empty_stub_model_name

    global interface

    # if a model exists - shut it down before trying to create the new one
    if interface is not None:
        interface.shutdown()
        del interface
        gc.collect()

    logging.info(f"Creating a new model [{new_model_name}]")
    if new_model_name == empty_stub_model_name:
        interface = EmptyStubInterface()
        interface.initialize()
    else:
        d = available_models[new_model_name]
        interface = LlamaOnnxInterface(
            onnx_file=d["onnx_file"],
            tokenizer_path=d["tokenizer_path"],
            is_GQA=d["is_GQA"],
        )
        interface.initialize()

    return new_model_name


def interface_predict(*args):
    global interface
    res = interface.predict(*args)

    for x in res:
        yield x


def interface_retry(*args):
    global interface
    res = interface.retry(*args)

    for x in res:
        yield x


with gr.Blocks(css=custom_css, theme=small_and_beautiful_theme) as demo:
    history = gr.State([])
    user_question = gr.State("")
    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="chuanhu_chatbot", height=900)
            with gr.Row():
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Enter text")
                with gr.Column(min_width=70, scale=1):
                    submit_button = gr.Button("Send")
                with gr.Column(min_width=70, scale=1):
                    cancel_button = gr.Button("Stop")
            with gr.Row():
                empty_button = gr.Button(
                    "🧹 New Conversation",
                )
                retry_button = gr.Button("🔄 Regenerate")
                delete_last_button = gr.Button("🗑️ Remove Last Turn")
        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="Parameter Setting"):
                    gr.Markdown("# Model")
                    model_name = gr.Dropdown(
                        choices=[empty_stub_model_name] + list(available_models.keys()),
                        label="Model",
                        show_label=False,  # default="Empty STUB",
                    )
                    model_name.change(
                        change_model_listener, inputs=[model_name], outputs=[model_name]
                    )

                    gr.Markdown("# Parameters")
                    top_p = gr.Slider(
                        minimum=-0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        interactive=True,
                        label="Top-p",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    max_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=512,
                        value=256,
                        step=8,
                        interactive=True,
                        label="Max Generation Tokens",
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=2048,
                        step=128,
                        interactive=True,
                        label="Max History Tokens",
                    )
    gr.Markdown(description)

    predict_args = dict(
        # fn=interface.predict,
        fn=interface_predict,
        inputs=[
            user_question,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    retry_args = dict(
        fn=interface_retry,
        inputs=[
            user_input,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ],
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )

    reset_args = dict(fn=reset_textbox, inputs=[], outputs=[user_input, status_display])

    # Chatbot
    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_question, user_input, submit_button],
        show_progress=True,
    )

    predict_event1 = user_input.submit(**transfer_input_args).then(**predict_args)

    predict_event2 = submit_button.click(**transfer_input_args).then(**predict_args)

    empty_button.click(
        reset_state,
        outputs=[chatbot, history, status_display],
        show_progress=True,
    )
    empty_button.click(**reset_args)

    predict_event3 = retry_button.click(**retry_args)

    delete_last_button.click(
        delete_last_conversation,
        [chatbot, history],
        [chatbot, history, status_display],
        show_progress=True,
    )
    cancel_button.click(
        cancel_outputing,
        [],
        [status_display],
        cancels=[predict_event1, predict_event2, predict_event3],
    )

    demo.load(change_model_listener, inputs=None, outputs=model_name)

demo.title = "Llama Chat UI"

demo.queue(concurrency_count=1).launch()
