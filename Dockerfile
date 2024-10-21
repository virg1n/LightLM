# Stage 1: Build the Flask backend
FROM python:3.9-slim as backend

# Set the working directory to the backend folder
WORKDIR /app

# Install Flask, Flask-CORS, torch, and any other necessary libraries
RUN pip install --upgrade pip
RUN pip install Flask Flask-CORS torch

# Copy backend code to the container
# This assumes your Dockerfile is in the root directory (TRYINGLLMA)
COPY ./website/backend /app

# Expose the port Flask will run on
EXPOSE 5501

# Command to run the Flask app
CMD ["python", "main.py"]


# Stage 2: Build the frontend
FROM nginx:alpine as frontend

# Remove the default Nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static files (index.html, scripts.js, styles.css) to Nginx directory
COPY ./website/index.html /usr/share/nginx/html/
COPY ./website/scripts.js /usr/share/nginx/html/
COPY ./website/styles.css /usr/share/nginx/html/

# Expose the default port for Nginx
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
