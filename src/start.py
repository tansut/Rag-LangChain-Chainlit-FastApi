from chat_app import ChatApp
import chainlit as cl

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

