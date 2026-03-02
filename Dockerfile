FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py ./

CMD ["python", "-u", "handler.py"]
