FROM python:3.12-slim
WORKDIR /flask_app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
WORKDIR /flask_app/flask_server
CMD ["flask", "run", "--host=0.0.0.0"]