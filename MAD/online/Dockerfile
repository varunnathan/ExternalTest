FROM python:3.8.3 as base
COPY ./pip-requirements.txt /recommend/
WORKDIR "/recommend"
RUN pip install -r pip-requirements.txt

FROM base as build
ENV PYTHONPATH=".:etc.environment"
COPY . /recommend/
EXPOSE 8880
WORKDIR "/recommend/src"

CMD python app.py
