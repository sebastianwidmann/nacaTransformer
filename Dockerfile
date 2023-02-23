FROM tensorflow/tensorflow:devel-gpu

# set the working directory in container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN python -m pip install -r requirements.txt

# copy the content of the local source directory to the working directory
COPY src/ .

# execute container
CMD [ "python3", "./main.py"]
