# Rock, Paper, Scissors, Lizard, Spock

This project introduces an extension to the traditional Rock, Paper, Scissors game by incorporating additional hand gestures: Lizard and Spock. The objective is to train a machine learning model directly in the browser using a webcam.

## How it Works

Participants train their model similarly to the Rock, Paper, Scissors example, but now include the Lizard and Spock hand gestures. The model is trained by collecting images for each hand gesture using the webcam. 

### Key Tips:
1. **Model Architecture:** Avoid overfitting by keeping the model architecture similar to the Rock, Paper, Scissors example.
2. **Data Collection:** Collect between 50 to 100 images for each hand gesture for model training.
3. **Training Completion:** Wait until training has finished before downloading the model. Check the console output for the "Training is Done!" alert and ensure LOSS values stop being printed before downloading the model.
4. **Model Evaluation:** Models are graded based on performance on the provided test set images.

## Skills Demonstrated
- Utilizing TensorFlow.js for in-browser machine learning model training.
- Image data collection and preprocessing for training purposes.
- Understanding and implementing neural network architectures for classification tasks.
- Grading model performance based on provided test set images.

## Preview
![Training Example](https://github.com/22bayusetia/22bayusetia/blob/main/Rock%2C%20Paper%2C%20Scissors/Preview%20RPS.png)





