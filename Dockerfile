FROM python:3.8

WORKDIR /sentimentanalysis

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download pt_core_news_lg

COPY . .

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0"]