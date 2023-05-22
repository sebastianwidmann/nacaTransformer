FROM python:3.11

# copy contents of src directory to workdir
COPY requirements.txt .
COPY src/ .

# install dependencies

# install GPU-version of JAX and FLAX
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax
RUN pip install optax
RUN pip install -q clu
RUN pip install tensorflow-datasets
RUN pip install -U matplotlib

RUN pip install -r requirements.txt

# execute container
CMD ["python3", "-m", "src.main", "--config=src/config.py"]
