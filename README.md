# Fashion-Recommendation

Overview
This project implements a fashion recommendation system using machine learning. It utilizes a pre-trained ResNet model for image feature extraction and Annoy for efficient nearest neighbor search based on image embeddings. The system aims to provide personalized fashion recommendations based on user-selected images.

Functionality
The system integrates image metadata, normalizes features, and uses transfer learning with ResNet for image feature extraction. It builds an Annoy index to quickly retrieve similar fashion items and generates recommendations based on user preferences and selected images.

Usage
The project requires Python 3.x and libraries such as matplotlib, numpy, pandas, fastai, annoy, and others. It involves acquiring fashion images and metadata, training the ResNet model, and using the Annoy index for recommendation generation.
