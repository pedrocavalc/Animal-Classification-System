FROM python:3.9.18-bullseye

WORKDIR app/

COPY . .
# update and install dependencies
RUN apt-get update && \
apt -y install python3-venv && \
python3 --version  &&\
python3 -m venv env  \
&& . env/bin/activate \
&& pip3 install -r requirements.txt
