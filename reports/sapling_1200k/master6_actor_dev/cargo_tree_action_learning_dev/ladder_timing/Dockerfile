FROM python:3.12-slim
COPY numpy /usr/local/lib/python3.12/site-packages/numpy
COPY numpy.libs /usr/local/lib/python3.12/site-packages/numpy.libs
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
