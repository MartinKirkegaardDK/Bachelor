FROM python:3.10.10
# Specify our path in the container
WORKDIR /bachelor

# We copy everything from this path
COPY . /bachelor/


RUN pip install --upgrade pip
# Here we install the dependencies
RUN pip install numpy
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install mlflow


#Here we run the script
CMD [ "python", "main_martin.py"]