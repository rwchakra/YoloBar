docker rmi -f $(docker images -aq)
docker build ./ -f Dockerfile -t submission1
docker run -v /home/ctm/Desktop/GitHub/visum-competition2022/data:/data submission1