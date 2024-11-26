FROM bitnami/pytorch:2.3.0

WORKDIR /app
USER root

RUN apt-get update
RUN apt-get install gcc python3-dev ffmpeg -y

COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY . .

CMD [ "python3", "app.py" ]