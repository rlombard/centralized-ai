# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies (including libraries needed for OpenCV and other packages)
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for the project
RUN python -m venv /opt/venv

# Upgrade pip
RUN /opt/venv/bin/pip install --upgrade pip

# Install the requirements from requirements.txt
RUN /opt/venv/bin/pip install -r requirements.txt

# Copy the models folder (already downloaded models)
COPY ./app/models /app/models

# Expose the application's port
EXPOSE 8000

# Set environment variables to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Run the app with uvicorn
CMD ["uvicorn", "app.ai_server:app", "--host", "0.0.0.0", "--port", "8000"]
