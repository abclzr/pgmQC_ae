# Base image with Python
FROM python:3.10-slim

# Install basic utilities (optional but often good)
RUN apt-get update && apt-get install -y \
    build-essential \
    graphviz \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your code into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# (Optional) default command
CMD [ "bash" ]
