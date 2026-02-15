This project is an end-to-end Machine Learning application that predicts flight ticket prices based on multiple features such as airline, source, destination, duration, and journey date.

It also demonstrates a complete MLOps pipeline, including model deployment, containerization, and CI/CD automation.

Project Architecture:
Data → Preprocessing → Model Training →
Model Serialization → Flask API →
Docker → Jenkins CI/CD → Deployment

Project Structure:
flight_price_app/
│── app.py
│── train.py
│── preprocessing.py
│── model.pkl
│── requirements.txt
│── Dockerfile
│── Jenkinsfile
│── README.md

