### Overview
Regarding the deployment aspect, it's worth noting that while Huggingface provides a convenient platform for serving models directly from the Model Hub, it might not always be the best fit for all use cases. The direct deployment from Huggingface's platform might have limitations or specific configurations that didn't align with the desired setup.
Switching to Docker Desktop provided more control over the deployment environment. Docker Desktop, combined with a manual setup, offers a more customizable and manageable solution, especially when considering local testing and development. Using Docker Desktop, one can easily manage containers, networks, and storage options via a graphical interface, which can be especially helpful during the development and testing phases.

I've provided the code to create a Docker container that runs an inference service using Flask, Gunicorn, NGINX, and a pretrained Huggingface model(Image Classifer). The service is designed to accept image files via a POST request, run inference using the Vision Transformer (ViT) model, and return the predicted class. I have also provided a Jupyter notebook code that demonstrates how to send an image to this service and print out the prediction.

### Components
1. **Dockerfile**: This is responsible for building the Docker image for the service.
2. **app.py**: The core Flask application that handles image processing and model prediction.
3. **nginx.conf**: The NGINX configuration that forwards requests to the Flask app.
4. **ML API Demonstration.ipynb**: A Jupyter notebook for demonstrating how to call the API and display the result.

### Instructions and Explanations:

#### Dockerfile:
This Dockerfile does the following:
1. Uses the `python:3.8-slim` image as the base image.
2. Sets the working directory to `/app`.
3. Installs NGINX and Python libraries like Flask, gunicorn, transformers, torch, and Pillow.
4. Copies the `app.py` file and the NGINX configuration to the appropriate locations.
5. Starts the NGINX service and runs the Flask app using Gunicorn when the container is started.

#### app.py:
This script:
1. Imports necessary libraries.
2. Loads the pretrained Vision Transformer model and tokenizer from Huggingface's model hub.
3. Defines a Flask route (`/predict`) that accepts POST requests containing an image, processes the image, runs inference, and returns the predicted class.

#### nginx.conf:
This is a basic configuration for NGINX that listens on port 80 and forwards requests to the Flask app running on port 8000. It also sets some proxy headers.

#### ML API Demonstration.ipynb:
This Jupyter notebook:
1. Imports necessary libraries.
2. Loads and displays an image.
3. Sends the image to the inference service.
4. Prints out the predicted class.


### Why this model?
The Vision Transformer (ViT) is a paradigm shift from the traditional Convolutional Neural Networks (CNNs) used in image classification. Instead of leveraging convolutional layers, the Vision Transformer uses the Transformer architecture (originally designed for NLP tasks) to process images. Here's why this model is particularly notable:

1. **Innovative Approach**: Unlike CNNs, which process images using convolutions, ViT divides images into fixed-size patches, linearly embeds them, and processes them as a sequence, much like how sentences are processed in NLP tasks.
2. **High Performance**: Despite its novelty, ViT has demonstrated competitive performance with state-of-the-art CNNs on image classification tasks, especially when trained on large datasets.
3. **Pretraining**: The model you're using is pretrained on ImageNet-21k and fine-tuned on ImageNet 2012. This dual-phase training allows the model to benefit from a vast amount of data and generalize well to other datasets.
4. **Flexibility**: Transformers provide a flexible architecture that can be adapted to a variety of tasks, and ViT showcases this flexibility in the domain of computer vision.



### Deployment Steps:
1. **Building the Docker image**: 
   ```bash
   docker build -t vit-classifier .
   ```
2. **Running the Docker container**: 
   ```bash
   docker run -p 80:80 vit-classifier
   ```
   
3. Once the container is running, you can use the provided Jupyter notebook to send an image to the local API endpoint (`http://localhost/predict`) and get the prediction.
