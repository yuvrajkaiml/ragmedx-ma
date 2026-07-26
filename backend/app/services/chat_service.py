from app.llm.local_llm import LocalLLMWrapper


class ChatService:

    def __init__(self):
        self.llm = LocalLLMWrapper()

    def generate_response(
        self,
        question: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        # temperature and max_tokens are ignored for now
        # They will be supported later when we refactor LocalLLMWrapper.

        return self.llm.generate( 
                                 prompt=question,
                                 temperature=temperature,
                                 max_new_tokens=max_tokens,
                                )