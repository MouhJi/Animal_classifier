FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /deeplearning/model

COPY model model


RUN apt-get update
RUN apt-get install vim -y
RUN pip install scikit-learn tensorboard opencv-python matplotlib
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip
RUN unzip model/dataset.zip


CMD ["python", "./model/train_model.py"]