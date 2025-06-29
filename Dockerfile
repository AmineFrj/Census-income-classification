FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

RUN apt-get update && apt-get install -y libgomp1

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token='1234567890'", "--NotebookApp.password='1234567890'"]
