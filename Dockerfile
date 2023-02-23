FROM python:3.8

# copy the dependencies file to the working directory
COPY requirements.txt .

# install requirements
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# run container
CMD [ "python", "./main.py"]
