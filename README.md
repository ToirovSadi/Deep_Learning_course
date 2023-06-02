# Table of contents
1. [Deep Learning School](#deep-learning-school)
2. [Information about assigments](#assignments-during-the-course)
3. [Task Desciption](#task-description)
    1. [Task1_ML_contest](#task1_ml_contest)
    2. [Task2_CNN](#task2_cnn)
    3. [Task3_Image_Classification](#task3_image_classification)


# Deep Learning School
## What is the Deep Learning School?

The Deep Learning School is an educational organization at the Applied Mathematics and Computer Science School of the Moscow Institute of Physics and Technology (MIPT). They offer courses on artificial intelligence for schoolchildren and students interested in programming and mathematics. This particular course is aimed at individuals who have significant programming experience in Python and are confident in their knowledge of school and university mathematics, as well as familiarity with data libraries (numpy, pandas, matplitlib).

The course covers the theory and practice of deep learning and neural networks in an interactive format. The course includes practical assignments and a final project. Upon completion of the course, students will receive a certificate that can give them an advantage when applying for bachelor's and master's degrees at the MIPT School of Applied Mathematics and Computer Science. The topics covered in the advanced level course include the general theory of neural networks and neural networks in computer vision.

Deep Learning School [link](https://dls.samcs.ru/en/dls), and [link](https://stepik.org/course/135003/) to stepik platform where the course is provided.

## Assignments during the course
### Format of the tasks
#### ! In each directory you will find at least two files, one named **'task_description.ipynb'** which is just **template** that was given in assignments by the instructors. The other file is solution !
<br>

## Task Description
### **Task1_ML_contest**
Your task is to learn how to model customer churn for a telecommunications company. This task is very important in practice, and algorithms for solving it are used in real 
telecommunications companies. If we know that a customer is going to leave us, we will try 
to retain them by offering some bonuses. On the data page, you can download two files - 
train.csv (contains feature columns and target variables) and test.csv (contains only 
feature columns). As an answer, you need to upload predictions of the probabilities of 
customer churn for test.csv. 
<br><br>


### **Task2_CNN**

The primary objective of the task was to initially employ ANN to classify images, followed by incorporating a CNN layer to assess its impact and experimenting with various activation functions. The initial subtask involved categorizing images into ten classes using the MNIST dataset, training a PyTorch model on the dataset, and exploring diverse activation functions. Subsequently, a Convolutional layer was integrated to evaluate the potent impact of CNN layers on image preprocessing, along with experimentation with kernels.


### **Task3_Image_Classification**

The whole task is to classify images from provided dataset in Kaggle. The dataset contains around 20000 images of simpsons cartoon. The goal is to train a model to predict a certain character from that cartoon. The dataset contains 42 classes. First I tried with a simple CNN model and then I used transfer learning with EfficientNet-b2 model. The results are shown in the notebook. With simple CNN my model get overfitting, but with transfer learning I got a good result in 30 epochs.