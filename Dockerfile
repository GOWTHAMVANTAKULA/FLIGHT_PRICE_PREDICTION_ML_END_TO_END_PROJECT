
# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only necessary files (thanks to .dockerignore)
COPY requirements.txt .
COPY app.py .
COPY *.joblib ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]