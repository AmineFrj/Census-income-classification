FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

RUN apt-get update && apt-get install -y libgomp1

CMD ["./start.sh"]
