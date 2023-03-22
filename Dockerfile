FROM tensorflow/tensorflow:latest-gpu

# set the working directory in container
WORKDIR /nacatf

# copy the content of the local source directory to the working directory
COPY requirements.txt .
COPY src/ .
COPY airfoilMNIST/ .

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev

# install dependencies
RUN python -m pip install -r requirements.txt

#export PYTHONPATH="${PYTHONPATH}:/src/"

# execute container
CMD [ "python3", "main.py"]
