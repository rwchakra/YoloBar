# Select Python Docker Image (v3.8)
FROM python:3.8


# Perform necessary updates to set up the Docker container
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Copy necessary directories to the Docker container
COPY ./code /code
COPY ./results/models/visum2022.pt results/models/visum2022.pt


# Install Python libraries
RUN pip3 install -U albumentations
RUN pip3 install torchinfo
RUN pip3 install tqdm
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu


# Run command to test the participants' model and generate the score
CMD ["python3", "/code/model_test.py"]
