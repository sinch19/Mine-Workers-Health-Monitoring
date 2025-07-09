# Real-Time Health Monitoring for Mine Workers Using AI and ML

## 📌 Project Overview

This project aims to enhance worker safety and optimize task allocation in the mining industry by predicting worker capability using health data collected from scheduled checkups. It uses machine learning to classify workers into three categories — **Best**, **Moderate**, or **Limited** — based on their health and wellness indicators.

## 🧠 Problem Statement

Mining is a high-risk industry where workers are frequently exposed to harsh environmental conditions. Traditional health monitoring methods are periodic, manual, and reactive. This project provides a **proactive, AI-based solution** that allows supervisors to make informed decisions regarding worker assignments based on predicted health status.

## ✅ Features

- 🔍 **Prediction of Worker Capability** (Best, Moderate, Limited)
- 📊 **Input Parameters**: Age, BMI, SpO₂ level, alcohol consumption, job satisfaction
- ⚙️ **AI-based Neural Network Model** for classification
- 🖥️ **User-friendly Interface** built using Streamlit
- ⚡ **Real-time Processing** of health data (no database storage as of now)
- 📌 **Task Recommendations** based on prediction

## 🛠️ Tools and Technologies Used

- **Python** — Programming language
- **NumPy, Pickle** — Data handling and model serialization
- **TensorFlow / Keras** — For building and training the neural network model
- **Scikit-learn** — For preprocessing (StandardScaler)
- **Streamlit** — Interface for data input and prediction display

## 🚀 How It Works

1. **User Input**: Supervisor enters health data into the interface.
2. **Preprocessing**: Data is normalized using StandardScaler.
3. **Prediction**: Trained neural network model predicts worker capability.
4. **Output**: The system displays capability status and recommendations.
5. **Reset Option**: Allows the supervisor to reset inputs and start over.
