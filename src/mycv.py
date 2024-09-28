from lib.rag import Rag, LLMS, EMBEDDINGS
from lib.core import ChatSettings
from chat_app import ChatApp
import chainlit as cl
chat_settings = ChatSettings()

from pydantic import BaseModel, Field
from typing import List, Union, Optional
from langchain_core.output_parsers import JsonOutputParser

class ResultWithFollowup(BaseModel):
    """Result with followup"""
    answer: str = Field(description="Answer to the question")
    follow_up_questions: Optional[List[str]] = Field(default_factory=list, description="Followup questions as list")


rag = ChatApp.rag = Rag(
    inputFolder="mycv",
    promptFile="mycv.txt",
    chat_settings=chat_settings,
    output_formatter = JsonOutputParser(pydantic_object=ResultWithFollowup)
)

rag.initialize_store()

ChatApp.starters = [
        cl.Starter(
            label="Who is Tansu",
            message="Who is Tansu",
            icon="/public/chat-bot-svgrepo-com.svg",
            )
        ]

ChatApp.use_followup = True

@cl.password_auth_callback
def auth_callback(username: str, password: str):
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )

