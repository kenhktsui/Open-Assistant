from typing import List, Optional
from pydantic import BaseModel


class ScorerOutput(BaseModel):
    """
    model_id corresponds to model id in HF, or your own model id
    score_id refers to unique id in this pipeline
    """
    model_id: str
    score_id: str
    score: float


class InstructAnswer(BaseModel):
    instruct: str
    answer: str
    score: Optional[List[ScorerOutput]]

    def __str__(self):
        return f"Prompt:\n{self.instruct}\nResponse:\n{self.answer}"

