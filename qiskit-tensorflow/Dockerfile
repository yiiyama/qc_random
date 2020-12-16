FROM nvcr.io/nvidia/tensorflow:20.11-tf2-py3

RUN pip install qiskit qiskit-aer-gpu numba && \
    pip cache purge

EXPOSE 8888

RUN useradd -ms /bin/bash quantum
USER quantum
WORKDIR /home/quantum
