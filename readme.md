## Table of Contents
  - [Artistify](#artistify)
  - [Table of Contents](#table-of-contents)
  - [Project Motivation](#project-motivation)
  - [Project Description](#project-description)
  - [Project Status: Completed](#project-status-completed)
  - [Getting Started](#getting-started)
  - [Additional features (can be implemented)](#additional-features-can-be-implemented)
  - [Major Technologies Used](#major-technologies-used)
  - [Libraries required](#libraries-required)


# Artistify
Artistify is a Deep Learning based simple web application, that helps demonstrate and simulate the concept of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer) in Deep Learning, using Tensorflow and Keras.


## Project Motivation
I was always curious about the various applications of deep learning to different aspects of our life. One of the most interesting applications is Neural Style Transfer. The basic concept of NST is to learn the style and features of a style image, and apply it to the content image to generate an artistic image. This project can demonstrate how well deep learning can achieve considerable success in the field of art and creativity.


## Project Description
The purpose of this project is to simulate the idea of Neural Style Transfer using Deep Learning, using Tensorflow(Python) with Keras, and flask for web server.


## Project Status: Completed


## Getting Started

1. Clone [this repo](https://github.com/romilshah525/artistify) (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Make sure all the necesary dependencies are installed, conda is preffered.
3. Go to the project folder and run 'flask run' or 'python3 main.py' in your terminal. The flask server will start at http://127.0.0.1:5000/, by defualt.
4. Upload content and style images, and monitor the progress in the track bar.
5. Once the image is generated, all the three images will be shown.


## Additional features (can be implemented)
- Share images via the socket.io connection as well.
- Make the generated image update as the training progresses.


## Major Technologies Used
* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Socket.io](https://socket.io/)


## Libraries required
* [Tensorflow](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Flask Socket.io](https://flask-socketio.readthedocs.io/en/latest/)
* [WSGI](https://wsgi.readthedocs.io/en/latest/)
* [PIL](https://pillow.readthedocs.io/en/stable/)
* [Numpy](https://numpy.org/)
