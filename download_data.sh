# Install gdown library to download stuff from Google Drive
pip install gdown

# Download "data.zip"
gdown https://drive.google.com/file/d/17f4M6rlPTq0sc-2acJvy-4gVSrDCL5rx/view?usp=sharing --fuzzy -O data.zip

# Unzip file to "data"
unzip data.zip -d data

# Remove "data.zip"
rm data.zip
