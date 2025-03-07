# Use official Python image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
