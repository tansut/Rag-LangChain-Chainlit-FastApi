
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from langchain_core.output_parsers import JsonOutputParser

class ResultWithFollowup(BaseModel):
    """Result with followup"""
    answer: str = Field(description="Answer to the question")
    follow_up_questions: Optional[List[str]] = Field(default_factory=list, description="Followup questions as list")

