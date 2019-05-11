FROM jjanzic/docker-python3-opencv

COPY requirements.txt /
RUN pip install -r /requirements.txt

WORKDIR /opt/webapp/

COPY . /opt/webapp/

RUN adduser --disabled-password myuser
USER myuser

CMD gunicorn --bind 0.0.0.0:$PORT wsgi 