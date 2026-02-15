# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run Flask app
CMD ["python", "app.py"]
