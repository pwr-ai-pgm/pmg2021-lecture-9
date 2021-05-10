FROM jupyter/base-notebook

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /lecture/lecture