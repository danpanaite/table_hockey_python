FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt