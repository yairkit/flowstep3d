#!/bin/bash

echo "Downloading models.."
mkdir checkpoints

# Download
fileId=1RpZVdGacp0Y4H145ekUtcssmrSOhVLqF
fileName=flowstep3d_checkpoints.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

# Extract
tar -xzvf ${fileName} -C checkpoints
rm ${fileName}