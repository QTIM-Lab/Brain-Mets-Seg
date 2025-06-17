ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.11-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install nvidia-pyindex
ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
RUN pip install tensorflow-addons --upgrade

# AWS Client for data downloading
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip -qq awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws

ENV OMP_NUM_THREADS=2
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV OMPI_MCA_coll_hcoll_enable 0
ENV HCOLL_ENABLE_MCAST 0 

WORKDIR /workspace/brain_mets_seg
ADD . /workspace/brain_mets_seg

# Download trained models from dropbox
RUN mkdir -p /workspace/brain_mets_seg/trained_models
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_0
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_1
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_2
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_3
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_4

RUN wget -O /workspace/brain_mets_seg/trained_models/Model_0/checkpoint https://www.dropbox.com/scl/fi/3i951l43jxy653bcurrz0/checkpoint?rlkey=mt23dsd3axjbadncabcdi0anf&st=zypk1njw&dl=0
RUN wget -O /workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.data-00000-of-00001 https://www.dropbox.com/scl/fi/n3kr6edpxvfw900tf080q/ckpt-best-978.data-00000-of-00001?rlkey=1lhl3y9z6u57n27vxtqwvh1g6&st=mb94841u&dl=0
RUN wget -O /workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.index https://www.dropbox.com/scl/fi/fc39vcxfdqp0owp445o67/ckpt-best-978.index?rlkey=bczxks3bczwe50g325f7yd021&st=mklcax7x&dl=0

COPY predict /usr/local/bin/predict
COPY entrypoint /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/predict /usr/local/bin/entrypoint

ENTRYPOINT ["entrypoint"]