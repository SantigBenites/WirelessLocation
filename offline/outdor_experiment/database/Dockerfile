# Use Ubuntu 22.04 as the base image (MongoDB 7.0 is supported on this version)
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update the package repository and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gnupg \
    ca-certificates \
    wget \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Import MongoDB public GPG key
RUN wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | apt-key add -

# Create a list file for MongoDB
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -sc)/mongodb-org/7.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Update and install MongoDB
RUN apt-get update && \
    apt-get install -y mongodb-org && \
    rm -rf /var/lib/apt/lists/*

# Create the data directory
RUN mkdir -p /data/db && \
    chown -R mongodb:mongodb /data/db

# Set the user to mongodb (better security practice than running as root)
USER mongodb

# Expose MongoDB port
EXPOSE 27017

# Set the default command to run when the container starts
CMD ["mongod", "--bind_ip_all"]