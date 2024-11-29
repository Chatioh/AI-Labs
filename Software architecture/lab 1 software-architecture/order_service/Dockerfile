# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

# set the working directory in the container
WORKDIR /app

#copy the service file and reqired dependency files
COPY order_service.py /app/order_service.py

# Install pip requirements
COPY requirements.txt /app/requirements.txt

# install dependencies
RUN  pip install fastapi uvicorn

# Export the port used by the application
EXPOSE 8000

# Command to run the service
CMD ["uvicorn", "order_service:app", "--host", "0.0.0.0", "--port", "8000"]
