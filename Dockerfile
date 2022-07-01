FROM python:3.8

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*



# COPY ./requirements.txt /requirements.txt

COPY ./code /code

COPY ./results/models/visum2022.pt results/models/visum2022.pt

# RUN pip3 install opencv-python-headless
RUN pip3 install -U albumentations
RUN pip3 install torchinfo
RUN pip3 install tqdm
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# RUN pip3 install --no-cache-dir -r /requirements.txt
# 

CMD ["python3", "/code/model_predictions.py"]
