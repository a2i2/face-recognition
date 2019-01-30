# Externalise base image so that we can build GPU and non-GPU images.
ARG BASE_IMAGE
FROM $BASE_IMAGE

# Set default locale.
ENV LANG C.UTF-8

# Install system packages/dependencies.
RUN apt-get --yes update && apt-get --yes install cmake libx11-dev libfontconfig1 libxrender1 libsm6 libxext6

# Install Python packages.
RUN pip3 install --upgrade pip
RUN pip3 install pytz==2018.3
RUN pip3 install tzlocal==1.5.1
RUN pip3 install PyYAML==3.12
RUN pip3 install piexif==1.1.0b0
RUN pip3 install pika==0.11.2
RUN pip3 install imgaug==0.2.5
RUN pip3 install six==1.11.0
RUN pip3 install stringcase==1.2.0
RUN pip3 install opencv-python==3.4.0.12
RUN pip3 install keras==2.1.5
RUN pip3 install psycopg2==2.7.4
RUN pip3 install boto3==1.5.8
RUN pip3 install flask==1.0.2

RUN pip3 install itsdangerous==0.24
RUN pip3 install Jinja2==2.9.6
RUN pip3 install MarkupSafe==1.0
RUN pip3 install Werkzeug==0.12.2

# Clean up to reduce image size.
RUN rm -rf /root/.cache/pip /var/lib/apt/lists/*

# Copy AI Models.
COPY ./src/main/resources/models/20170512-110547/ /opt/theia/src/main/resources/models/20170512-110547/
COPY ./src/main/resources/models/Multi-task-CNN/ /opt/theia/src/main/resources/models/Multi-task-CNN
COPY ./src/main/resources/models/image-rotator/ /opt/theia/src/main/resources/models/image-rotator

# Copy code and config.
COPY ./src/main/resources/config /opt/theia/src/main/resources/config
COPY ./src/main/python /opt/theia/src/main/python

RUN cd ../opt/theia/src/main/python/surround	;	python setup.py install;
