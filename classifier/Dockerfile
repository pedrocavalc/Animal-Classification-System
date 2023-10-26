FROM tensorflow/tensorflow:2.12.0-gpu
WORKDIR /app
COPY . .

RUN pip3 install -r requirements.txt 

EXPOSE 5000

ENTRYPOINT [ "python3", "main.py" ]