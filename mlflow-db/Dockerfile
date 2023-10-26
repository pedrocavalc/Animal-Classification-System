FROM tensorflow/tensorflow:2.12.0-gpu
WORKDIR /app
COPY . .

RUN apt-get update && \
pip3 install -r requirements.txt 

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "file:///mlflow-artifacts"]