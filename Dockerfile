# 1. Start from a lightweight Python 3.14 image
FROM python:3.14-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy ONLY the requirements file
COPY requirements.txt .

# 4. Install dependencies
#    (we do this before copying the code for caching)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Now copy the rest of your code into the container
COPY . .

# 6. Default command (optional, but good practice)
CMD ["python", "code2.py"]
