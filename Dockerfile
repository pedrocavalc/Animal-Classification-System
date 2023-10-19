FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app
COPY requirements.txt .
# update and install dependencies
RUN apt-get update && \
pip3 install -r requirements.txt && \
apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

WORKDIR /app/src/app

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "manager:app", "--host", "0.0.0.0" , "--port", "8000" ]