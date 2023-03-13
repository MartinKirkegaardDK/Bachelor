FROM python:latest
# Specify our path in the container
WORKDIR /bachelor

# We copy everything from this path
COPY . /bachelor/

# Here we install the dependencies
RUN pip install numpy
RUN pip install pandas
RUN pip install -U scikit-learn



#Here we run the script
CMD [ "python", "./"]