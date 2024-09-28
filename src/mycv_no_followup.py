from lib.rag import Rag, LLMS, EMBEDDINGS
from lib.core import ChatSettings
from chat_app import ChatApp
import chainlit as cl
chat_settings = ChatSettings()

from pydantic import BaseModel, Field
from typing import List, Union, Optional
from langchain_core.output_parsers import JsonOutputParser

rag = ChatApp.rag = Rag(
    inputFolder="mycv",
    promptFile="mycv_no_followup.txt",
    chat_settings=chat_settings
)

rag.initialize_store()

ChatApp.starters = [
        cl.Starter(
            label="Who is Tansu",
            message="Who is Tansu",
            icon="/public/chat-bot-svgrepo-com.svg",
            )
        ]

@cl.password_auth_callback
def auth_callback(username: str, password: str):
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )

