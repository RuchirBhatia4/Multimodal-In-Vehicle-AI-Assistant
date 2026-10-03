from __future__ import annotations

import faulthandler
import logging
import signal
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from drivemind.config import settings
from drivemind.server.models import models
from drivemind.server.session import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
# Native crashes (segfaults in C extensions) print every thread's Python stack instead of
# dying silently, and `kill -USR1 <pid>` dumps all stacks on demand to debug a hang.
faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, all_threads=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=models.load_all, daemon=True, name="model-loader").start()
    yield


app = FastAPI(title="DriveMind", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")


@app.get("/")
async def index():
    return FileResponse(settings.web_dir / "index.html")


@app.get("/health")
async def health():
    return {"models": models.status, "load_times_s": models.load_times}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    await session.start()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.on_bytes(msg["bytes"])
            elif msg.get("text") is not None:
                import json

                await session.on_json(json.loads(msg["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        session.close()


def main() -> None:
    import uvicorn

    uvicorn.run("drivemind.server.app:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
