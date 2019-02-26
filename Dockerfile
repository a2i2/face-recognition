FROM ubuntu:16.04

# Set default locale.
ENV LANG C.UTF-8

RUN apt-get --yes update && apt-get --yes install python3-pip cmake libx11-dev libfontconfig1 libxrender1 libsm6 libxext6 libglib2.0-0

# Install Python packages.
RUN pip3 install --upgrade pip
RUN pip3 install surround==0.0.2
RUN pip3 install opencv-python==3.4.1.15
RUN pip3 install numpy==1.14.5
RUN pip3 install tensorflow==1.8.0
RUN pip3 install tornado==5.1.1
RUN pip3 install psycopg2==2.7.7
RUN pip3 install imutils==0.5.2

# Clean up to reduce image size.
RUN rm -rf /root/.cache/pip /var/lib/apt/lists/*

# Copy code and config.
COPY . /usr/local/src/a2i2/face-recognition

WORKDIR /usr/local/src/a2i2/face-recognition
ENTRYPOINT ["python3", "-m", "face-recognition"]
CMD []
