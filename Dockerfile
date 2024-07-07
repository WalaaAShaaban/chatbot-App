FROM python:3.10
RUN mkdir /app

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501

CMD streamlit run src/app.py