# Making denoising CNNs more generalisable by adding diverse unsupervised data #
# Project for Advanced Graphics and Image Processing #
This research analyses the utilisation of unsupervised data for making denoising networks more generalisable. The aim of these experiments is to make the model more robust without needing to rely on training on new supervised image pairs. Specifically, the experiment analyses the use of Transformation Consistency Regularization on unlabelled data by adding these samples to the training set and testing the performance improvement versus not augmenting the data in image denoising tasks.

## Demo and instructions
Examples of loading the data, training and validating the models can be seen in the Colab notebook. The Colab notebook with the demo can be found in the following URL: https://colab.research.google.com/drive/1kaL17UXvdw_YvEq6EML6BQVb5zF2FT9n?usp=sharing

## Trained Models
https://drive.google.com/drive/folders/1ZiQcpziQ86NFLrUV1KsJHzfGz1nhK2Y4?usp=sharing

## Supporting repositories
Original FFDNet repository: http://www.ipol.im/pub/art/2019/231/
Original TCR repository: https://github.com/aamir-mustafa/Transformation-CR