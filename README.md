# Mood Music Detector 🎵

## Overview

Mood Music Detector is an intelligent web-based application that analyzes a user's facial expression through a webcam and predicts their emotional state. Based on the detected mood, the system suggests suitable music to enhance the user's experience.

## Features

* Detects facial expressions using a live webcam feed
* Uses a Convolutional Neural Network (CNN) to classify emotions
* Recommends music according to the identified mood
* Interactive web interface built with Flask

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* Flask
* HTML, CSS, and JavaScript

## Project Structure

```
app.py                 → Backend application using Flask  
collect_data.py        → Script for collecting facial landmark data  
convert_to_csv.py      → Converts the collected dataset into CSV format  
train_model.py         → Trains the emotion recognition model  
templates/index.html   → Frontend user interface  
requirements.txt       → List of required Python libraries  
```

## Steps to Run the Project

1. Activate the virtual environment

```
venv\Scripts\activate
```

2. Gather facial emotion data

```
python collect_data.py
```

3. Convert the collected data into CSV format

```
python convert_to_csv.py
```

4. Train the machine learning model

```
python train_model.py
```

5. Launch the web application

```
python app.py
```

Open your browser and visit:

```
http://localhost:5000
```

## Future Enhancements

* Expand the system to recognize more emotions
* Enable continuous real-time emotion tracking
* Integrate music streaming services such as Spotify
* Improve prediction accuracy with larger datasets and model tuning
