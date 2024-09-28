import chainlit as cl  
from pydantic import BaseModel, Field

def get_chat_profiles():
    return [    
        cl.ChatProfile(
            name="OpenAI",
            markdown_description="OpenAI GPT",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="Antrophic",
            markdown_description="Antrophic GPT",
            icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="Cohere",
            markdown_description="Cohere GPT",
            icon="https://picsum.photos/250",
        ),
        cl.ChatProfile(
            name="Ollama",
            markdown_description="Ollama GPT",
            icon="https://picsum.photos/250",
        )
    ]