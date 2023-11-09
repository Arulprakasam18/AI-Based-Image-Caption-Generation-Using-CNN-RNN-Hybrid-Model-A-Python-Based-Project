# AI-Based-Image-Caption-Generation-Using-CNN-RNN-Hybrid-Model-A-Python-Based-Project
# Image Caption Generator – Python based Project

This project aims to create an AI-based image caption generator using Python. The system combines Convolutional Neural Networks (CNN) for feature extraction from images and Long Short-Term Memory (LSTM) networks for generating descriptive captions.

## Understanding CNN

Convolutional Neural Networks (CNN) are specialized deep neural networks designed to process data represented as 2D matrices, making them ideal for working with images. CNNs excel in tasks such as image classification and identifying objects in images.

## Working of Deep CNN

A deep CNN scans images from left to right and top to bottom, extracting essential features from the image. These features are then combined to classify images. Deep CNNs can handle translated, rotated, scaled, and perspective-transformed images.

## What is LSTM?

LSTM stands for Long Short-Term Memory, a type of Recurrent Neural Network (RNN) suitable for sequence prediction tasks. LSTM networks are effective at predicting the next word in a sequence based on previous text. They overcome the short-term memory limitations of traditional RNNs and maintain relevant information while discarding non-relevant data using a forget gate.

## Image Caption Generator Model

To build the image caption generator, we merge the CNN and LSTM architectures, creating a CNN-RNN model.

- CNN is used to extract features from images, utilizing the pre-trained Xception model.
- LSTM uses information from the CNN to generate image descriptions.

## Project File Structure

The project has a specific file structure:

- **Downloaded from dataset:**
  - `Flicker8k_Dataset`: A dataset folder containing 8091 images.
  - `Flickr_8k_text`: A dataset folder with text files containing image captions.

- **Files created by us:**
  - `Models`: This directory contains the trained models.
  - `Descriptions.txt`: A text file containing all image names and their preprocessed captions.
  - `Features.p`: A pickle object containing image feature vectors extracted from the Xception pre-trained CNN model.
  - `Tokenizer.p`: A pickle object that maps tokens to index values.
  - `Model.png`: A visual representation of the project's architecture.
  - `Testing_caption_generator.py`: A Python file for generating captions for any image.
  - `Training_caption_generator.ipynb`: A Jupyter notebook used for training and building the image caption generator.

## Building the Python-based Project

The project follows these key steps:

1. **Data Collection and Preprocessing:** Data, including images and captions, is collected and preprocessed, handling missing values and outliers.
2. **Feature Extraction:** The Xception model is used to extract image features, and these are stored in the `Features.p` file.
3. **Loading Dataset for Training:** 6000 training images are loaded, along with their corresponding captions and features.
4. **Tokenizing the Vocabulary:** The vocabulary is tokenized, mapping words to index values, and saved in the `Tokenizer.p` file.
5. **Creating Data Generator:** A data generator yields input and output sequences in batches for training.
6. **Defining the CNN-RNN Model:** The model is created using Keras' Functional API, comprising a feature extractor, sequence processor, and decoder.
7. **Training the Model:** The model is trained on the 6000 training images and their captions, with the results saved in the `models` directory.
8. **Testing the Model:** A separate Python file, `Testing_caption_generator.py`, loads the trained model and generates captions for given images.

