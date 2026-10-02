from collections.abc import Sequence
from functools import cached_property
from typing import Protocol

from tokenizers import Tokenizer

# paraphrase-multilingual-MiniLM-L12-v2 truncates at 128 tokens including [CLS] and [SEP].
# See docs/adr/0004-small-passages-for-multilingual-embedder.md.
EMBEDDER_WINDOW = 128
MAX_PASSAGE_TOKENS = EMBEDDER_WINDOW - 2


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def count_tokens(self, text: str) -> int: ...


class FastEmbedEmbedder:
    dim = 384

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding(self.model_name)

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        tok = Tokenizer.from_pretrained(self.model_name)
        tok.no_truncation()
        return tok

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
