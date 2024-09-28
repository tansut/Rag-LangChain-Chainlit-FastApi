from langchain.schema.runnable.config import RunnableConfig
import uuid
import chainlit as cl
import os
from chainlit.input_widget import Select, Switch, Slider
from lib.cl_llms import get_chat_profiles
from lib.rag import Rag, LLMS, UptatableChatHistory
from lib.core import ChatSettings
from langchain_core.language_models import LanguageModelLike
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Union, Optional

class ChatApp:
    
    rag: Rag
    starters = []
    use_followup = False
    
    @cl.on_chat_start
    async def on_chat_start():
        pass
    
    @cl.set_starters
    async def set_starters():
        return ChatApp.starters
    
    @cl.on_settings_update
    async def setup_agent(settings):
        ChatApp.rag.chat_settings.temperature = settings["temperature"]
        ChatApp.rag.chat_settings.top_p = settings["top_p"]
        
    @cl.set_chat_profiles
    async def chat_profile():
        return get_chat_profiles()

    async def stream(message: cl.Message | str, runnable: LanguageModelLike, chat_history: UptatableChatHistory):
        response = cl.Message("")
        if isinstance(message, str): message = cl.Message(id=uuid.uuid4().hex, content=message)
        
        chat_history.check_message_update(message.id)
        
        async for chunk in runnable.astream(
            {"input": message.content, "user": cl.user_session.get("user").identifier, "chat_history": chat_history.messages},
            config=RunnableConfig(configurable={ "session_id":cl.user_session.get("session_id"),  "message_id": "message.id"}, callbacks=[cl.LangchainCallbackHandler()]),
        ):
            has_answer = chunk and "answer" in chunk
            await response.stream_token(token=chunk["answer"] if has_answer else "", is_sequence=True)
        
        chat_history.add_message(message=HumanMessage(content=message.content, id=message.id))
        chat_history.add_ai_message(message=AIMessage(content=chunk["answer"], id=uuid.uuid4().hex))
        
        await response.send()
        
        follow_ups = chunk["follow_up_questions"]
        actions = []
        
        if (follow_ups and len(follow_ups) > 0):
        
            for fq in follow_ups:
                actions.append(cl.Action(fq,fq, f"{fq}"))
            
            actions.append(cl.Action("","","✅ I have question"))
            
            fa = await cl.AskActionMessage("", actions=actions).send()
            if fa and fa.get("value") != "":
                await cl.Message(content=fa.get("label"), type="user_message").send()
                await ChatApp.stream(fa.get("label"), runnable, chat_history)
        
    @cl.on_chat_start
    async def on_chat_start():
        chat_profile = cl.user_session.get("chat_profile") or "openai"
        chat_history = UptatableChatHistory()
        
        await cl.ChatSettings(
            [
                Slider(
                    id="temperature",
                    label="Temperature",
                    initial=ChatApp.rag.chat_settings.temperature,
                    min=0,
                    max=1,
                    step=0.1,
                ),
                Slider(
                    id="top_p",
                    label="TopP",
                    initial=ChatApp.rag.chat_settings.top_p,
                    min=0,
                    max=1,
                    step=0.1,
                )
            ]
        ).send()
        
        try:
            llm = ChatApp.rag.create_llm(LLMS[chat_profile.upper()])
            runnable = ChatApp.rag.create_runnable(llm)
            cl.user_session.set("llm", llm)  
            cl.user_session.set("runnable", runnable)
            cl.user_session.set("chat_history", chat_history)
            cl.user_session.set("session_id", f"{chat_profile}/{uuid.uuid4().hex}")
            if os.getenv("CONTEXTUALIZATION") == "True": 
                ChatApp.rag.contextualize_llm = llm 
            else: 
                ChatApp.rag.contextualize_llm = None
        except Exception as ex:
            await cl.Message(f"Please ensure you have set API keys in .env file").send()
            print(ex)
        
    @cl.on_message
    async def on_message(message: cl.Message):
        llm: LanguageModelLike = cl.user_session.get("llm")  
        chat_history = cl.user_session.get("chat_history")  
        try:
            llm.temperature = ChatApp.rag.chat_settings.temperature
            llm.top_p = ChatApp.rag.chat_settings.top_p
        except Exception as e:
            print(e)
        
        runnable = cl.user_session.get("runnable")  
        try:
            await ChatApp.stream(message, runnable, chat_history)
        except Exception as ex:
            await cl.Message(f"An error occured while processing your message.\n{ex}").send()
            print(ex)
        


