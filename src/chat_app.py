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
import os
from io import BytesIO
import base64
import httpx
from lib.tts import text_to_speech, speech_to_text
from io import BytesIO
from typing import IO
from openai import AsyncOpenAI
from chainlit.element import ElementBased
from langchain_core.output_parsers import JsonOutputParser
from lib.follow_up import ResultWithFollowup

openai_client = AsyncOpenAI()

class ChatApp:
    starters = []
    
    def get_rag() -> Rag:
        return cl.user_session.get("rag", None)
    
    def create_rag():
        chat_settings = ChatApp.get_rag().chat_settings if ChatApp.get_rag() else ChatSettings()
        rag: Rag
        if cl.user_session.get("follow_up"):
            rag = Rag(
                inputFolder="mycv",
                promptFile="mycv.txt",
                chat_settings=chat_settings,
                output_formatter = JsonOutputParser(pydantic_object=ResultWithFollowup)
            )
        else:
            rag = Rag(
                inputFolder="mycv",
                promptFile="mycv_no_followup.txt",
                chat_settings=chat_settings
            )
        rag.initialize_store()
        cl.user_session.set("rag", rag) 
        chat_profile = cl.user_session.get("chat_profile") or "openai"
        llm = rag.create_llm(LLMS[chat_profile.upper()])
        runnable = rag.create_runnable(llm)
        cl.user_session.set("llm", llm)  
        cl.user_session.set("runnable", runnable)
        if os.getenv("CONTEXTUALIZATION") == "True": 
            rag.contextualize_llm = llm 
        else: 
            rag.contextualize_llm = None
        return rag
    

    @cl.on_audio_chunk
    async def on_audio_chunk(chunk: cl.AudioChunk):
        if chunk.isStart:
            buffer = BytesIO()
            buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
            cl.user_session.set("audio_buffer", buffer)
            cl.user_session.set("audio_mime_type", chunk.mimeType)
        cl.user_session.get("audio_buffer").write(chunk.data)

    
    @cl.on_audio_end
    async def on_audio_end(elements: list[ElementBased]):
        audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
        audio_buffer.seek(0)  
        audio_file = audio_buffer.read()
        audio_mime_type: str = cl.user_session.get("audio_mime_type")

        input_audio_el = cl.Audio(
            mime=audio_mime_type, content=audio_file, name=audio_buffer.name
        )

        whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
        transcription = await ChatApp.stt(whisper_input)
        
        await cl.Message(
            author="You", 
            type="user_message",
            content=transcription,
            elements=[input_audio_el, *elements]
        ).send()
        
        runnable = cl.user_session.get("runnable")  
        chat_history = cl.user_session.get("chat_history")  
        
        await ChatApp.stream(transcription, runnable, chat_history)
        

    @cl.step(type="tool")
    async def stt(audio_file):
        return await speech_to_text(audio_file)

    
    @cl.step(type="tool")
    def tts(text: str, mime_type: str = "audio/webm", voice_id = None) -> IO[bytes]:
        return text_to_speech(text=text, mime_type=mime_type, voice_id=voice_id)
    
    @cl.action_callback("followup_button")
    async def on_action(action):
        runnable = cl.user_session.get("runnable")  
        chat_history = cl.user_session.get("chat_history")  
        await cl.Message(content=action.value, type="user_message").send()
        await ChatApp.stream(action.value, runnable, chat_history)
    
    
    @cl.set_starters
    async def set_starters():
        return ChatApp.starters
    
    @cl.on_settings_update
    async def setup_agent(settings):
        ChatApp.get_rag().chat_settings.temperature = settings["temperature"]
        ChatApp.get_rag().chat_settings.top_p = settings["top_p"]
        cl.user_session.set("tts", settings["tts"])
        if (cl.user_session.get("follow_up", True) != settings["follow_up"]):
            cl.user_session.set("follow_up", settings["follow_up"])
            ChatApp.create_rag()
        
        
    @cl.set_chat_profiles
    async def chat_profile():
        return get_chat_profiles()

    async def stream(message: cl.Message | str, runnable: LanguageModelLike, chat_history: UptatableChatHistory):
        response = cl.Message("")
        if isinstance(message, str): message = cl.Message(id=uuid.uuid4().hex, content=message)
        
        chat_history.check_message_update(message.id)
        
        ai_answer = ''
        
        async for chunk in runnable.astream(
            {"input": message.content, "user": cl.user_session.get("user").identifier, "chat_history": chat_history.messages},
            config=RunnableConfig(configurable={ }, callbacks=[cl.LangchainCallbackHandler()]),
        ):
            if cl.user_session.get("follow_up"):
                has_answer = chunk and "answer" in chunk
                ai_answer = chunk["answer"] if has_answer else ""
                await response.stream_token(token=ai_answer, is_sequence=True)
            else:
                ai_answer = ai_answer + chunk
                await response.stream_token(token=chunk)
                
        
                
        chat_history.add_message(message=HumanMessage(content=message.content, id=message.id))
        chat_history.add_ai_message(message=AIMessage(content=ai_answer, id=uuid.uuid4().hex))
        
        await response.send()
        
        if cl.user_session.get("tts"):
        
            audio_mime_type = "audio/webm"
            
            output_name, output_audio = ChatApp.tts(ai_answer, audio_mime_type, os.getenv("ELEVENLABS_VOICE_ID"))
            
            output_audio_el = cl.Audio(
                name=output_name,
                auto_play=True,
                mime=audio_mime_type,
                content=output_audio,
            )
            

            response.elements = [output_audio_el]
            await response.update()           
            
        if cl.user_session.get("follow_up"):    
            current_actions = []
            
            follow_ups =  chunk["follow_up_questions"] if "follow_up_questions" in chunk else None
            
            if (follow_ups and len(follow_ups) > 0):
                for fq in follow_ups:
                    current_actions.append(cl.Action(name="followup_button", label=fq, value=fq))
                cl.user_session.set("current_actions", current_actions)
                response.actions = current_actions
                await response.update()   
                #await cl.Message(content="", actions=current_actions).send()
                
    @cl.on_chat_start
    async def on_chat_start():
        try:
            await ChatApp.ensure_initialize_chat()
        except Exception as ex:
            await cl.Message(f"Please ensure you have set API keys in .env file").send()
            print(ex)
            

        
            
    async def ensure_initialize_chat():
        cl.user_session.set("tts", False)
        cl.user_session.set("follow_up", True)
        chat_profile = cl.user_session.get("chat_profile") or "openai"
        chat_history = UptatableChatHistory()     
        cl.user_session.set("chat_history", chat_history)
        cl.user_session.set("session_id", f"{chat_profile}/{uuid.uuid4().hex}")
        rag = ChatApp.create_rag()
    
        await cl.ChatSettings(
        [
            Switch(id="follow_up", label="Followup Questions", initial=cl.user_session.get("follow_up")),
            Switch(id="tts", label="Use TTS", initial=cl.user_session.get("tts")),
            Slider(
                id="temperature",
                label="Temperature",
                initial=ChatApp.get_rag().chat_settings.temperature,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="top_p",
                label="TopP",
                initial=ChatApp.get_rag().chat_settings.top_p,
                min=0,
                max=1,
                step=0.1,
            )
        ]
    ).send()            

        
    @cl.on_message
    async def on_message(message: cl.Message):
        
        llm: LanguageModelLike = cl.user_session.get("llm")  
        chat_history = cl.user_session.get("chat_history")  
        try:
            llm.temperature = ChatApp.get_rag().chat_settings.temperature
            llm.top_p = ChatApp.get_rag().chat_settings.top_p
        except Exception as e:
            print(e)
        
        
        runnable = cl.user_session.get("runnable")  
        
        try:
            await ChatApp.stream(message, runnable, chat_history)
        except Exception as ex:
            await cl.Message(f"An error occured while processing your message.\n{ex}").send()
            print(ex)
        


