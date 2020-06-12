FROM rocm/tensorflow:rocm3.3-tf2.1-dev

LABEL maintainer="Shogo Takata <peshogo@gmail.com>"

COPY docker/Pipfile /tmp/Pipfile

RUN pip3 install --upgrade pip && pip3 install pipenv

RUN cd /tmp && LC_ALL=C.UTF-8 LANG=C pipenv lock -r > requirements.txt && pip3 install requirements.txt

