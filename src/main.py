from fastapi import FastAPI, Body, APIRouter
from fastapi.responses import RedirectResponse
import uvicorn
from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

load_dotenv()

def run_application():
    uvicorn.run(app="main:create_app",
            factory=True,
            reload=True)

def create_app() -> FastAPI:
    app = FastAPI();
    mount_chainlit(app=app, target="src/chainlit_start.py", path="/chat")
    
    @app.get("/")
    def read_root():
        return RedirectResponse("/chat")
    
    return app


if __name__ == "__main__":
    run_application()