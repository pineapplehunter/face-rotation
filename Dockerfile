FROM rocm/tensorflow:rocm3.5-tf2.2-dev

LABEL maintainer="Shogo Takata <peshogo@gmail.com>"

RUN pip3 install --upgrade pip && pip3 install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml /tmp/pyproject.toml
COPY poetry.lock /tmp/poetry.lock

RUN cd /tmp &&\
    poetry add tensorflow-rocm &&\
    poetry remove tensorflow &&\
    poetry update

WORKDIR /root