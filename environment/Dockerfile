FROM registry.codeocean.com/codeocean/miniconda:4.3.30-python2.7.14-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "build-essential=12.1ubuntu2" \
      "gcc=4:5.3.1-1ubuntu1" \
      "xorg=1:7.7+13ubuntu3.1" \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN /postInstall
