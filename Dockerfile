FROM python:3.5

ADD ./AI-Bootcamp-Demo /AI-Bootcamp-Demo
WORKDIR /AI-Bootcamp-Demo

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python","main.py"]