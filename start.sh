#!/bin/bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=1234567890 --NotebookApp.password=1234567890 &
uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
