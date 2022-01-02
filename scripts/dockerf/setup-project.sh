#!/bin/sh

echo "Downloadig docker image..."
docker build . --progress tty -t udacity/mlops:latest
