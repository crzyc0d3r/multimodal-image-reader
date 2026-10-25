"""Local-image question answering and OCR over Ollama's chat API."""

from .client import Conversation, ImageReaderError, OllamaClient

__all__ = ["Conversation", "ImageReaderError", "OllamaClient"]
