from app_modules.utils import logging


class EmptyStubInterface:
    def __init__(self):
        pass

    def initialize(self):
        pass

    def shutdown(self):
        pass

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
        logging.info("hi there")
        logging.info("-" * 100)

        history.append([text, f"[Empty Model] You typed: {text}"])
        yield history, history, "No Model Found"
        # yield [[text, "No Model Found"]], [], "No Model Found"

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
        yield chatbot, history, "Empty context"
