FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /

RUN apt-get update && apt-get install -y \
        ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender-dev libx11-xcb1 \
        gcc g++ libvips-dev --no-install-recommends

COPY docker/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workdir