# Use an official Ubuntu as the base image
FROM ubuntu:latest

USER $USER

# Update the package repository
RUN apt-get update

# Install necessary packages
RUN apt-get install -y gnupg lsb-release

# Import the public key used by the package management system
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 160D26BB1785BA38

# Create a list file for MongoDB
RUN sh -c 'echo "deb http://repo.mongodb.org/apt/ubuntu $(lsb_release -sc)/mongodb-org/7.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list'

# Update the package repository
RUN apt-get update 

# Install MongoDB
RUN apt-get install -y mongodb-org

# Create the data directory
RUN mkdir -p /data/db

# Set the default command to run when the container starts
CMD ["mongod", "--bind_ip_all","--quiet"]