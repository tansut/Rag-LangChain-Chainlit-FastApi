import chainlit as cl  
from pydantic import BaseModel, Field

class ChatSettings(BaseModel):
    temperature: float = Field(0.7)
    top_p: float = Field(1)