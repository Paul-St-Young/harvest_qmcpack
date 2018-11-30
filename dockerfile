FROM gcc:8.2.0
RUN apt-get update &&\
    apt-get install -y git &&\
    apt-get install -y libhdf5-dev &&\
    apt-get install -y python-pip
RUN git clone https://github.com/Paul-St-Young/harvest_qmcpack.git
WORKDIR /harvest_qmcpack
RUN pip install -r requirements.txt
ENV PYTHONPATH=/harvest_qmcpack:$PYTHONPATH
CMD pytest -v .
