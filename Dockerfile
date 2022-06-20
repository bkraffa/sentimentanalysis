FROM python:3.8

WORKDIR /sentimentanalysis

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn model_api:app --host 0.0.0.0 --port $PORT"]

#docker build . -t sentimentanalysis --no-cache
#docker run -p 8000:4000 -e PORT=4000 sentiment