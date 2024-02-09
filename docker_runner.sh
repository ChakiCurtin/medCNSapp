#!/bin/bash

# - [ This is what i have used to get this application working on docker[very new to porting application to docker] ] - #

# [Make sure docker has sudo access (group access) ]
# https://docs.docker.com/engine/install/linux-postinstall/#:~:text=If%20you%20don't%20want,members%20of%20the%20docker%20group.

docker build . -f dockerfile -t medstreamapp
docker run -dp 127.0.0.1:8501:8501 medstreamapp

# RUN WITH NVIDIA enabled
docker run -p 127.0.0.1:8501:8501 --rm --runtime=nvidia medstreamapp

# if you want it attached:
# docker run -p 127.0.0.1:8501:8501 medstreamapp