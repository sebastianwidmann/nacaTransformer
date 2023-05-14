FROM python:3.11

# copy contents of src directory to workdir
COPY requirements.txt .
COPY src/ .

# install dependencies

# install GPU-version of JAX and FLAX
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax

RUN pip install -r requirements.txt

# execute container
CMD ["python3", "main.py"]