# Implementing state-of-the-art networks for the game of Go


* Authors: Neveu Pierre
* Course: Deep Learning
* University: PSL - Paris Dauphine

**Please see DL_Report.pdf where I explain my work**

## Context

The goal of this project is to train a network for playing the game of Go. Each week, **a round robin tournament** will be organized with students networks. Each network will be used by a **PUCT engine** that takes 2 seconds of CPU time at each move to play in the tournament. In order to be fair with the training, the networks submitted must be around ** 100,000 parameters**.  

The data used for training comes from the **Katago Go program** self played games. There are 1 000 000 different games in total in the training set. The **input data** is composed of **31 19x19 planes** (color to play, ladders, current state on two planes, two previous states on four planes). The **output targets** are the policy (a vector of size 361 with 1.0 for the move played, 0.0 for the other moves), and the value (close to 1.0 if White wins, close to 0.0 if Black wins).

## Environment
This project was developped with a Google Colab Notebook and Google Drive in 01/01/2025.

## Source code
Games and data implementations are handled by a C++ code (/Game folder) provided by Tristan Cazenave. This projects focuses on building a network in a supervised-learning manner.

The source code for the project is also available on the website of Tristan Cazenave
(i.e. the teacher of the deep learning course), you can import the code in the colab environment by importing the following command. Warning, files
will not be saved if you close your notebook.

```python
!wget https://www.lamsade.dauphine.fr/~cazenave/project2025.zip
!unzip project2025.zip
```
OR
```python
path_to_project = '/content/drive/MyDrive/Deep_Learning-Go_Project/Game' # Update with your path

from google.colab import drive
import os

# Import google drive
drive.mount('/content/drive')

# Navigate to the project directory.
os.chdir(path_to_project)
!ls
```

## Tensorflow version
This project was developped under tensorflow 2.15.0.

```python
!pip install tensorrt-bindings==8.6.1
!pip install --extra-index-url https://pypi.nvidia.com tensorrt-libs
!pip install tensorflow[and-cuda]==2.15.0
```
