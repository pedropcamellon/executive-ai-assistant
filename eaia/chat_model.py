from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from os import getenv
import vertexai


class ChatModel:
    def __init__(
        self,
        model_provider: Literal["openai"] | Literal["anthropic"] | Literal["google"],
        model_name: str,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.model_name = model_name

        # Validate
        if model_provider in ["openai", "anthropic", "google"]:
            self.model_provider = model_provider
        else:
            raise ValueError(f"Unknown model provider: {model_provider}")

    def get_model(self):
        """Get the model based on the model provider"""
        # TODO Add more model providers

        if self.model_provider == "openai":
            return self.openai_model()
        elif self.model_provider == "anthropic":
            return self.anthropic_model()
        elif self.model_provider == "google":
            return self.google_model()
        else:
            raise ValueError(f"Unknown model provider: {self.model_provider}")

    def openai_model(self):
        return ChatOpenAI(
            model=self.model_name,
            **self.kwargs,
        )

    def anthropic_model(self):
        return ChatAnthropic(
            model=self.model_name,  # Example: "claude-3-5-sonnet-latest"
            **self.kwargs,
        )

    def google_model(self):
        vertexai.init(
            project=getenv("GCP_PROJECT_ID"),
            location=getenv("GCP_PROJECT_LOCATION"),
        )

        return ChatVertexAI(
            model=self.model_name,  # Example: "gemini-1.5-pro"
            **self.kwargs,
        )
