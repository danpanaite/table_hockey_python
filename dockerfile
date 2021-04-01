FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY requirements.txt ./

# COPY hockey_scraper_data ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt