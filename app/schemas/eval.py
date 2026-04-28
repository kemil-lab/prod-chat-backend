from pydantic import BaseModel
from typing import List


class EvalInputRow(BaseModel):
    question: str
    reference: str


class EvalOutputRow(BaseModel):
    user_input: str
    response: str
    reference: str
    retrieved_contexts: List[str]