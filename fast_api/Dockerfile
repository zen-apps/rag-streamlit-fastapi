FROM python:3.9.18-slim

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r ./requirements.txt && rm -rf pip3 cache dir

WORKDIR /app
COPY app ./app/

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
