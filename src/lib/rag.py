from langchain_openai import ChatOpenAI
from typing import Any, Dict, Union
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere
import faiss
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os
from enum import Enum
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import LanguageModelLike
from lib.core import ChatSettings
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from operator import itemgetter
from typing import List, Union, Optional
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser, BaseLLMOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, BasePromptTemplate, format_document

LLMS = Enum("LLMS", ["OPENAI", "ANTROPHIC", "COHERE", "OLLAMA"])
EMBEDDINGS = Enum("EMBEDDINGS", ["openai", "cohere", "huggingface"])

class UptatableChatHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def check_message_update(self, message_or_id: BaseMessage | str):
        id = message_or_id.id if isinstance(message_or_id, BaseMessage) else message_or_id
        existing = next((x for x in self.messages if x.id == id), None)
        if existing:
            index = self.messages.index(existing)
            del self.messages[-(len(self.messages) - index):] 

    def add_message(self, message: BaseMessage) -> None:
        self.check_message_update(message)
        return super().add_message(message)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

class Rag:
    def __init__(self, inputFolder: str, promptFile: str, output_formatter: BaseLLMOutputParser = StrOutputParser(), embedding: EMBEDDINGS = EMBEDDINGS.openai, contextualize_prompt: str = None, structured_output = None,  chat_settings: ChatSettings = ChatSettings()):
        inputFiles = os.listdir(f"rag_source/{inputFolder}")
        self.inputFiles = list(map(lambda x: os.path.abspath(f"rag_source/{inputFolder}/{x}"), inputFiles))
        with open(f"prompt/{promptFile}", "r") as file:
            prompt = file.read()
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.structured_output = structured_output
        self.embedding = embedding
        self.chat_settings = chat_settings
        self.inputFolder =  inputFolder
        self.output_formatter = output_formatter
        self.llm_functions = {
            LLMS.OPENAI: ChatOpenAI,
            LLMS.ANTROPHIC: ChatAnthropic,
            LLMS.COHERE: ChatCohere,
            LLMS.OLLAMA: ChatOllama,
        }
        self.contextualize_prompt = contextualize_prompt or (
            """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is.""" 
        )
        
        self.contextualize_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "Latest Question: {input}")
            ]
        )
        
        self.contextualize_llm = None

        
    def get_embedding(self) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
        if self.embedding == EMBEDDINGS.openai:
            return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.embedding == EMBEDDINGS.huggingface:
            model_kwargs = {'device': 'mps'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=os.getenv("HUGGINGFACE_EMBED_MODEL"),
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            return hf
        
    def create_vector_store(self) -> FAISS:
        embedding = self.get_embedding()
        index = faiss.IndexFlatL2(len(embedding.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding,
            docstore=InMemoryDocstore(),
            index=index,
            index_to_docstore_id={}
        )
        
        return vector_store
    
    def initialize_store(self) -> FAISS:
        embedding = self.get_embedding()
        dir = f"{os.getenv("VECTOR_STORE_PATH")}/{self.inputFolder}/{self.embedding.name}"
        chunkSize: int = 2000
        chunkOverlap: int = 400
        self.store: FAISS = None
        if os.path.exists(dir):
            self.store = store = FAISS.load_local(
                dir, embedding, allow_dangerous_deserialization=True
            )
            print(f"Loading from local store {dir}")
        else:
            loaders = [PyPDFLoader(f"{file}") for file in self.inputFiles]
            documents = []
            for loader in loaders:
                documents.extend(loader.load())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
            splits = text_splitter.split_documents(documents)
            uuids = [str(uuid4()) for _ in range(len(splits))]
            self.store = self.create_vector_store()
            self.store.add_documents(documents=splits, ids=uuids)
            self.store.save_local(dir)
            print(f"Saved to local store {dir}")
            
        
    def create_runnable(self, llm: LanguageModelLike) -> RunnableWithMessageHistory:   
        
        def format_docs(inputs: dict) -> str:
            return "\n\n".join(
                format_document(doc, PromptTemplate.from_template("{page_content}")) for doc in inputs["context"]
            )
        
        def ensureContextualize(input_: dict):
            retriever = RunnableLambda(lambda input: self.store.similarity_search(input, k=4))
            if self.contextualize_llm is None or input_.get("chat_history") is None or len(input_.get("chat_history")) == 0:
                return RunnablePassthrough().assign(context=itemgetter("input") | retriever)
            else:
                return  ( self.contextualize_template 
                        | self.contextualize_llm 
                        | RunnableLambda(lambda x: input_ | {"input_contextual": x.content}) 
                        | RunnablePassthrough().assign(context=itemgetter("input") | retriever)
                        )
                
        rag_chain = ( RunnableLambda(ensureContextualize).with_config({"run_name":"ContextualizationCheck"}) 
                    | RunnablePassthrough.assign(context = format_docs).with_config({"run_name":"QueryDocuments"}) 
                    | self.prompt_template 
                    | llm 
                    | self.output_formatter
                    )
        
        return rag_chain
    

    def create_llm(self, llm_type: LLMS, model: str = None, chat_settings: ChatSettings = None) -> LanguageModelLike:
        model = model if model else os.getenv(f"{llm_type.name.upper()}_MODEL")
        chat_settings = chat_settings if chat_settings else self.chat_settings
        args = {
           "streaming": True,
           "model": model,
           "api_key": os.getenv(f"{llm_type.name.upper()}_API_KEY"),
           "temperature": chat_settings.temperature,
           "top_p": chat_settings.top_p,
        }

        
        llm = self.llm_functions[llm_type](**args)
        llm = llm if self.structured_output == None else llm.with_structured_output(self.structured_output)
        return llm
    