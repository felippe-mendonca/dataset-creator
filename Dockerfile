FROM ubuntu:16.04

ADD . /dataset-creator
WORKDIR /dataset-creator
RUN bash bootstrap.sh