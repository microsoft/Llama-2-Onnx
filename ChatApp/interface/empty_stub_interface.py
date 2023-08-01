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
        # yield chatbot,history,"Empty context."
        yield [[text, "No Model Found"]], [], "No Model Found"

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
