1. Web using: FastAPI + WebSocket
 - run with debugmode : uvicorn server:app --host 127.0.0.1 --port 8000 --reload --log-level debug
 - run with production mode : uvicorn server:app --host 0.0.0.0 --port 8000 --reload
2. kill port 8000
netstat -ano | findstr :8000
