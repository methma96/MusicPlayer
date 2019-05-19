FROM jjanzic/docker-python3-opencv

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN apt-get update
RUN apt-get install -y ffmpeg

WORKDIR /opt/webapp/

COPY . /opt/webapp/

RUN adduser --disabled-password myuser
RUN chmod -R a+rwx /opt/webapp/

USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT wsgi 