---
jupyter:
  accelerator: GPU
  colab:
  gpuClass: standard
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.2
  nbformat: 4
  nbformat_minor: 1
---

::: {.cell .markdown id="Giq4o2k_0xAI"}
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
:::

::: {.cell .code id="2gnJDBUEvHy7"}
``` python
from google.colab import drive
drive.mount('/content/drive')
```
:::

::: {.cell .markdown id="TY-cJ-0-1gTH"}
1.  Copy the zip file to Colab. This helps with speeding up image
    loading
:::

::: {.cell .code execution_count="19" id="lOyntkEMvFBV"}
``` python
!cp '/content/drive/MyDrive/TensorFlow-2.x-YOLOv3-master.zip' '.'
```
:::

::: {.cell .markdown id="E4xAoDji1r4u"}
1.  Unzip the file.
:::

::: {.cell .code id="rdS4_C9EwKQt"}
``` python
!unzip "TensorFlow-2.x-YOLOv3-master.zip"
```
:::

::: {.cell .markdown id="L_6hVXU93Pgg"}
1.  Change directories to access the model.
:::

::: {.cell .code execution_count="30" id="asACZW0Jwdcj"}
``` python
import os
from google.colab import drive
os.chdir('/content/TensorFlow-2.x-YOLOv3-master')
```
:::

::: {.cell .markdown id="k4smwfor15-X"}
1.  Download the required libraries. Most libraries needed are listed in
    \'requirements.txt\'.
:::

::: {.cell .code id="pyItpInv4liQ"}
``` python
!pip install -r requirements.txt  # install
```
:::

::: {.cell .markdown id="2TxZKWTi6_4l"}
1.  Upload all images needed for training to the directories
    \"/mnist/mnist_test\" and \"/mnist/mnist_train\".

2.  Save the classes\' names in a file named \"mnist.names\", with each
    class on a single line.

3.  Add the images\' paths (path to the \"/mnist/mnist_test\" directory)
    into the files \"mnist_test.txt\" and \"mnist_train.txt\", with each
    path on a single line (columns: image_path, xmin, ymin, xmax, ymax,
    class).

4.  Configure your model (in the file \"yolov3/configs.py\"). I saved my
    model on Google Drive (since Colab keeps disconnecting), so you may
    want to change the value of \'TRAIN_CHECKPOINTS_FOLDER\' to
    \"checkpoints\".

5.  Train the model (or retrain it) using:
:::

::: {.cell .code id="XtwrHvTKwbMc"}
``` python
!python3 train.py
```
:::

::: {.cell .markdown id="5X8_V9jJ8SEr"}
1.  Once the model is ready, use it to detect the images you want by
    providing the path of the images in a file named \'test.csv\'.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ExZWcZPpXK_t" outputId="53d20793-c79b-404d-b53a-1ca5c6a921b6"}
``` python
!python3 detect.py
```

::: {.cell .markdown id="WjPNj9mx8i5j"}
The results are ready in the file \'result0.txt\' in the required format
(just add the first row with the column names).

To save your current work, compress the working directory and move it to
your Google Drive.
:::

::: {.cell .code id="lR4UtmzvabgX"}
``` python
!zip -r 'TensorFlow-2.x-YOLOv3-master.zip' '../TensorFlow-2.x-YOLOv3-master'
!cp 'TensorFlow-2.x-YOLOv3-master.zip' '/content/drive/MyDrive'
```
:::

::: {.cell .markdown id="iRzk8huFoskn"}
By [Muhannad A.
Alwarawreh](https://www.linkedin.com/in/muhannad-alwarawreh-11045b222/)
:::
