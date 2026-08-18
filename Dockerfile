# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglpk-dev \
    glpk-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure PYTHONPATH includes the src directory
ENV PYTHONPATH="/app/src"

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit application using shell to evaluate $PORT
CMD sh -c "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true"
