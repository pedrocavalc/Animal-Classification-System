FROM tensorflow/tensorflow:2.12.0-gpu
WORKDIR /app

RUN apt-get update && \
apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

ENTRYPOINT [ "uvicorn", "manager:app", "--host", "0.0.0.0" , "--port", "8000" ]