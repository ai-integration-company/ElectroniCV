FROM python:3.9-slim

WORKDIR /src/web-app

COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Make the entrypoint script executable
RUN chmod +x /src/web-app/entrypoint.sh

# Run the entrypoint script
ENTRYPOINT ["/src/web-app/entrypoint.sh"]
