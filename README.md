# Visual Pollution Detection Model (Yolov4)

At the beginning, I read the task and analyzed the problem at hand.
Then, I started by simply downloading the images and writing a simple
Python code to view the images and rectangles (look
\"tools/viewImages.py\"). This helped me to see that the coordinates
needed rescaling (\*2).

All images\' sizes were reduced to (1/4) to speed up both the training
and detection of the model, and the coordinates were changed accordingly
(look \"tools/resizeImages.py\").

After that, I wrote a simple Keras Sequential model and added the VGG16
as a base model. Then, I started to use the YOLOv3 model.

The base model/files were imported from
[Here](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) and then
modified for this problem, first using YOLOv3 and then YOLOv4.

To help the YOLOv4 model read data from files, it was necessary to
generate the files \'mnist_train\' and \'mnist_test\' (which was done
using the script \"tools/csv2txt.py\").

After training the model on the given dataset, I found that it was not
entirely accurate, so I regenerated the dataset by making the model
predict the classes and removed any data with low probability.

One note, one of the classes had only one image and I found out about
that too late, so it was not possible to fix this issue. One of the
solutions for this problem would be to search for a free-to-use dataset
that contains this class and use it in the training process.

\"If you just want to test the current model, just skip to step 13.\"

How to use the model:

1.  Upload the zip file to Google Drive.
2.  Mount the drive with the zip file to Google Colab.

``` python
from google.colab import drive
drive.mount('/content/drive')
```

3.  Copy the zip file to Colab. This helps with speeding up image
    loading

``` python
!cp '/content/drive/MyDrive/TensorFlow-2.x-YOLOv3-master.zip' '.'
```

4.  Unzip the file.

``` python
!unzip "TensorFlow-2.x-YOLOv3-master.zip"
```

5.  Change directories to access the model.

``` python
import os
from google.colab import drive
os.chdir('/content/TensorFlow-2.x-YOLOv3-master')
```

6.  Download the required libraries. Most libraries needed are listed in
    \'requirements.txt\'.

``` python
!pip install -r requirements.txt  # install
```

7.  Upload all images needed for training to the directories
    \"/mnist/mnist_test\" and \"/mnist/mnist_train\".

8.  Save the classes\' names in a file named \"mnist.names\", with each
    class on a single line.

9.  Add the images\' paths (path to the \"/mnist/mnist_test\" directory)
    into the files \"mnist_test.txt\" and \"mnist_train.txt\", with each
    path on a single line (columns: image_path, xmin, ymin, xmax, ymax,
    class).

10.  Configure your model (in the file \"yolov3/configs.py\"). I saved my
    model on Google Drive (since Colab keeps disconnecting), so you may
    want to change the value of \'TRAIN_CHECKPOINTS_FOLDER\' to
    \"checkpoints\".

11.  Train the model (or retrain it) using:

``` python
!python3 train.py
```

12.  Once the model is ready, use it to detect the images you want by
    providing the path of the images in a file named \'test.csv\'.

``` python
!python3 detect.py
```

The results are ready in the file \'result0.txt\' in the required format
(just add the first row with the column names).

To save your current work, compress the working directory and move it to
your Google Drive.

``` python
!zip -r 'TensorFlow-2.x-YOLOv3-master.zip' '../TensorFlow-2.x-YOLOv3-master'
!cp 'TensorFlow-2.x-YOLOv3-master.zip' '/content/drive/MyDrive'
```

By Muhannad A. Alwarawreh
Alwarawreh](https://www.linkedin.com/in/muhannad-alwarawreh-11045b222/)
