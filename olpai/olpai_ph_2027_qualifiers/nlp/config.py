from dataclasses import dataclass

from common.config import BaseConfig


@dataclass
class Config(BaseConfig):
    model_name: str = "Qwen/Qwen2.5-1.5B"
    n_shuffles: int = 32
    batch_size: int = 128
    n_folds: int = 5
    function_words: tuple = ("the", "a", "an", "my", "our", "with", "of", "at", "in", "on", "before", "during",
                             "after", "to", "and", "from", "for", "by", "under", "over", "near", "beside")
    start_words: tuple = ("the", "a", "an", "my", "our")
    local_data: str = "data/nlp"
