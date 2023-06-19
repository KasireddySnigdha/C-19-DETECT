# C-19-DETECT
This GitHub repo contains code for accurate COVID-19 detection using chest X-ray images. It helps diagnose and monitor COVID-19 cases, aiding healthcare professionals.


WHAT IS DONE

Data Preparation:
The code starts by uploading the required dataset and extracting it.
The chest X-ray images are organized into three classes: "covid," "normal," and "virus."
Images from each class are loaded, resized to (224, 224), and stored in a data array along with their corresponding labels.

Data Split:
The data array and labels are split into training and testing sets using the train_test_split function.
The labels are converted into one-hot encoded vectors using tf.keras.utils.to_categorical.

Model Architecture:
The code defines a convolutional neural network (CNN) model using the Sequential API from TensorFlow.
The model consists of several convolutional layers with activation functions, max pooling layers, and fully connected layers.
The last layer uses the softmax activation function to classify the input into one of the three classes: COVID-19, normal, or virus.
The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy metric.

Model Training:
The model is trained on the training data for a specified number of epochs.
The training loss and accuracy are monitored and stored for later analysis.

Evaluation:
The trained model is evaluated on the testing data.
Classification metrics such as precision, recall, and F1-score are computed using the classification_report function.
A confusion matrix is generated using the confusion_matrix function to assess the model's performance.

Transfer Learning (Optional):
The code includes an optional section for transfer learning using the MobileNet architecture.
The MobileNet model is loaded with pre-trained weights from the ImageNet dataset.
The last few layers of the MobileNet model are frozen, and additional fully connected layers are added for classification.
The model is then recompiled, trained, and evaluated.

Model Visualization:
The code visualizes the training and validation loss as well as the training and validation accuracy using Matplotlib.

Gradio Integration (Optional):
The code installs the Gradio library for creating a user interface.
The input is an image, and the output is a label predicting the class of the input image.


DATA SET USED: CHEST X-RAY FROM KAGGLE(http://surl.li/dffxA)
