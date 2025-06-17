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

# Model Fold 0
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_0
RUN curl "https://www.dropbox.com/scl/fi/3i951l43jxy653bcurrz0/checkpoint?rlkey=mt23dsd3axjbadncabcdi0anf&st=zypk1njw&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/checkpoint"
RUN curl "https://www.dropbox.com/scl/fi/n3kr6edpxvfw900tf080q/ckpt-best-978.data-00000-of-00001?rlkey=1lhl3y9z6u57n27vxtqwvh1g6&st=mb94841u&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.data-00000-of-00001"
RUN curl "https://www.dropbox.com/scl/fi/fc39vcxfdqp0owp445o67/ckpt-best-978.index?rlkey=bczxks3bczwe50g325f7yd021&st=mklcax7x&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.index"

# Model Fold 1
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_1
RUN curl "https://www.dropbox.com/scl/fi/g39cdzw43i8tsbbqdyhv7/checkpoint?rlkey=ki8z8erjszvw8p4b21tm4jjsb&st=m5vob3q6&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/checkpoint"
RUN curl "https://www.dropbox.com/scl/fi/jr3159u1s2z0gocj45qya/ckpt-best-974.data-00000-of-00001?rlkey=166i8w8zp8qojurqqm5nw613v&st=wkg4o68h&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.data-00000-of-00001"
RUN curl "https://www.dropbox.com/scl/fi/9u1u9549pr2q4kgpzhm9q/ckpt-best-974.index?rlkey=mliu0ac1ietcg58kut5ttvwkn&st=ob7gtdmr&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.index"

# Model Fold 2
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_2
RUN curl "https://www.dropbox.com/scl/fi/93biajhxj07z74glh4cuo/checkpoint?rlkey=lcut88qc86hitw8mc70xahu10&st=4xkh3gcw&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/checkpoint"
RUN curl "https://www.dropbox.com/scl/fi/8jxti5esuy5rv7al7eu24/ckpt-best-986.data-00000-of-00001?rlkey=y2il7bcpi69zhfbebd3vehqie&st=6j7acico&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.data-00000-of-00001"
RUN curl "https://www.dropbox.com/scl/fi/qt34l0h2qja6fq6xb11cj/ckpt-best-986.index?rlkey=bsmjig1o23rq01csdif426avx&st=9nlxglrz&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.index"

# Model Fold 3
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_3
RUN curl "https://www.dropbox.com/scl/fi/90c3unaosxvknlmw62s8z/checkpoint?rlkey=08s2v9hev9ywt3kakuw326tay&st=rse8rszf&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/checkpoint"
RUN curl "https://www.dropbox.com/scl/fi/8u0qexg5qxf548zmqxfy8/ckpt-best-966.data-00000-of-00001?rlkey=evldhxl05ijs0ng2apkvskyp1&st=p8xipj4u&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.data-00000-of-00001"
RUN curl "https://www.dropbox.com/scl/fi/rix3n5v55j3aujpymly4d/ckpt-best-966.index?rlkey=brtclv3qsm32hu4v20l3wz5n7&st=iskn593j&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.index"

# Model Fold 4
RUN mkdir -p /workspace/brain_mets_seg/trained_models/Model_4
RUN curl "https://www.dropbox.com/scl/fi/3hk9purhdy3eee8l4dorv/checkpoint?rlkey=zb5t31ffwmc73untv110z71tk&st=ixvdonw3&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/checkpoint"
RUN curl "https://www.dropbox.com/scl/fi/cv6ar0yan8ljvjvglihok/ckpt-best-980.data-00000-of-00001?rlkey=b2c3nzu93jhnpebsqenhy2b22&st=0yfgbu9x&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.data-00000-of-00001"
RUN curl "https://www.dropbox.com/scl/fi/maor96gwcn2e9rfxdlzbc/ckpt-best-980.index?rlkey=fdzotevqmduhvynnmk0h7vnv1&st=y9zefkhb&dl=0" -o "/workspace/brain_mets_seg/trained_models/Model_0/ckpt-best-978.index"

CMD python predict.py