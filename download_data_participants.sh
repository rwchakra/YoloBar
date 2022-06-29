# Install gdown library to download stuff from Google Drive
pip install gdown

# Download "data.zip"
gdown https://drive.google.com/file/d/1gPhzIsvITrcFvITRy4H79EJ5VyXdxlTV/view?usp=sharing --fuzzy -O data_participants.zip

# Unzip file to "data"
unzip data_participants.zip -d data_participants

# Remove "data.zip"
rm data_participants.zip
