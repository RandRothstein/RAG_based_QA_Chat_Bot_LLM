# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for documents (relative to WORKDIR /app)
RUN mkdir -p documents

# Copy the core RAG logic file
COPY core_rag.py .

# Copy the entire ui directory
COPY ui/ ui/

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Command to run your Streamlit application from the 'ui' directory
# Ensure Python can find core_rag.py from ui/app.py
# Adding /app to PYTHONPATH so core_rag.py is discoverable
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]