#!/bin/bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_server:app