FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN apt update && apt install -y bash \
                   git \
                   curl

RUN apt install -y python3.10 \
                   python3-pip

RUN python3.10 -m pip install einops \
                icecream \
                flask \
                ruamel.yaml \
                uvicorn \
                fastapi \
                markdown2 \
                gradio \
                sconf \
                tensorboardX \
                tensorboard \
                h5py \
                sentencepiece \
                peft \
                opencv-python \
                decord \
                chardet \
                cchardet \
                huggingface-hub

RUN python3.10 -m pip install transformers==4.32.0