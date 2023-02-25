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

::: {.output .stream .stdout}
    2023-01-21 13:43:18.670474: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:42] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
    Start
    1 / 2092
    2 / 2092
    3 / 2092
    4 / 2092
    5 / 2092
    6 / 2092
    7 / 2092
    8 / 2092
    9 / 2092
    10 / 2092
    11 / 2092
    12 / 2092
    13 / 2092
    14 / 2092
    15 / 2092
    16 / 2092
    17 / 2092
    18 / 2092
    19 / 2092
    20 / 2092
    21 / 2092
    22 / 2092
    23 / 2092
    24 / 2092
    25 / 2092
    26 / 2092
    27 / 2092
    28 / 2092
    29 / 2092
    30 / 2092
    31 / 2092
    32 / 2092
    33 / 2092
    34 / 2092
    35 / 2092
    36 / 2092
    37 / 2092
    38 / 2092
    39 / 2092
    40 / 2092
    41 / 2092
    42 / 2092
    43 / 2092
    44 / 2092
    45 / 2092
    46 / 2092
    47 / 2092
    48 / 2092
    49 / 2092
    50 / 2092
    51 / 2092
    52 / 2092
    53 / 2092
    54 / 2092
    55 / 2092
    56 / 2092
    57 / 2092
    58 / 2092
    59 / 2092
    60 / 2092
    61 / 2092
    62 / 2092
    63 / 2092
    64 / 2092
    65 / 2092
    66 / 2092
    67 / 2092
    68 / 2092
    69 / 2092
    70 / 2092
    71 / 2092
    72 / 2092
    73 / 2092
    74 / 2092
    75 / 2092
    76 / 2092
    77 / 2092
    78 / 2092
    79 / 2092
    80 / 2092
    81 / 2092
    82 / 2092
    83 / 2092
    84 / 2092
    85 / 2092
    86 / 2092
    87 / 2092
    88 / 2092
    89 / 2092
    90 / 2092
    91 / 2092
    92 / 2092
    93 / 2092
    94 / 2092
    95 / 2092
    96 / 2092
    97 / 2092
    98 / 2092
    99 / 2092
    100 / 2092
    101 / 2092
    102 / 2092
    103 / 2092
    104 / 2092
    105 / 2092
    106 / 2092
    107 / 2092
    108 / 2092
    109 / 2092
    110 / 2092
    111 / 2092
    112 / 2092
    113 / 2092
    114 / 2092
    115 / 2092
    116 / 2092
    117 / 2092
    118 / 2092
    119 / 2092
    120 / 2092
    121 / 2092
    122 / 2092
    123 / 2092
    124 / 2092
    125 / 2092
    126 / 2092
    127 / 2092
    128 / 2092
    129 / 2092
    130 / 2092
    131 / 2092
    132 / 2092
    133 / 2092
    134 / 2092
    135 / 2092
    136 / 2092
    137 / 2092
    138 / 2092
    139 / 2092
    140 / 2092
    141 / 2092
    142 / 2092
    143 / 2092
    144 / 2092
    145 / 2092
    146 / 2092
    147 / 2092
    148 / 2092
    149 / 2092
    150 / 2092
    151 / 2092
    152 / 2092
    153 / 2092
    154 / 2092
    155 / 2092
    156 / 2092
    157 / 2092
    158 / 2092
    159 / 2092
    160 / 2092
    161 / 2092
    162 / 2092
    163 / 2092
    164 / 2092
    165 / 2092
    166 / 2092
    167 / 2092
    168 / 2092
    169 / 2092
    170 / 2092
    171 / 2092
    172 / 2092
    173 / 2092
    174 / 2092
    175 / 2092
    176 / 2092
    177 / 2092
    178 / 2092
    179 / 2092
    180 / 2092
    181 / 2092
    182 / 2092
    183 / 2092
    184 / 2092
    185 / 2092
    186 / 2092
    187 / 2092
    188 / 2092
    189 / 2092
    190 / 2092
    191 / 2092
    192 / 2092
    193 / 2092
    194 / 2092
    195 / 2092
    196 / 2092
    197 / 2092
    198 / 2092
    199 / 2092
    200 / 2092
    201 / 2092
    202 / 2092
    203 / 2092
    204 / 2092
    205 / 2092
    206 / 2092
    207 / 2092
    208 / 2092
    209 / 2092
    210 / 2092
    211 / 2092
    212 / 2092
    213 / 2092
    214 / 2092
    215 / 2092
    216 / 2092
    217 / 2092
    218 / 2092
    219 / 2092
    220 / 2092
    221 / 2092
    222 / 2092
    223 / 2092
    224 / 2092
    225 / 2092
    226 / 2092
    227 / 2092
    228 / 2092
    229 / 2092
    230 / 2092
    231 / 2092
    232 / 2092
    233 / 2092
    234 / 2092
    235 / 2092
    236 / 2092
    237 / 2092
    238 / 2092
    239 / 2092
    240 / 2092
    241 / 2092
    242 / 2092
    243 / 2092
    244 / 2092
    245 / 2092
    246 / 2092
    247 / 2092
    248 / 2092
    249 / 2092
    250 / 2092
    251 / 2092
    252 / 2092
    253 / 2092
    254 / 2092
    255 / 2092
    256 / 2092
    257 / 2092
    258 / 2092
    259 / 2092
    260 / 2092
    261 / 2092
    262 / 2092
    263 / 2092
    264 / 2092
    265 / 2092
    266 / 2092
    267 / 2092
    268 / 2092
    269 / 2092
    270 / 2092
    271 / 2092
    272 / 2092
    273 / 2092
    274 / 2092
    275 / 2092
    276 / 2092
    277 / 2092
    278 / 2092
    279 / 2092
    280 / 2092
    281 / 2092
    282 / 2092
    283 / 2092
    284 / 2092
    285 / 2092
    286 / 2092
    287 / 2092
    288 / 2092
    289 / 2092
    290 / 2092
    291 / 2092
    292 / 2092
    293 / 2092
    294 / 2092
    295 / 2092
    296 / 2092
    297 / 2092
    298 / 2092
    299 / 2092
    300 / 2092
    301 / 2092
    302 / 2092
    303 / 2092
    304 / 2092
    305 / 2092
    306 / 2092
    307 / 2092
    308 / 2092
    309 / 2092
    310 / 2092
    311 / 2092
    312 / 2092
    313 / 2092
    314 / 2092
    315 / 2092
    316 / 2092
    317 / 2092
    318 / 2092
    319 / 2092
    320 / 2092
    321 / 2092
    322 / 2092
    323 / 2092
    324 / 2092
    325 / 2092
    326 / 2092
    327 / 2092
    328 / 2092
    329 / 2092
    330 / 2092
    331 / 2092
    332 / 2092
    333 / 2092
    334 / 2092
    335 / 2092
    336 / 2092
    337 / 2092
    338 / 2092
    339 / 2092
    340 / 2092
    341 / 2092
    342 / 2092
    343 / 2092
    344 / 2092
    345 / 2092
    346 / 2092
    347 / 2092
    348 / 2092
    349 / 2092
    350 / 2092
    351 / 2092
    352 / 2092
    353 / 2092
    354 / 2092
    355 / 2092
    356 / 2092
    357 / 2092
    358 / 2092
    359 / 2092
    360 / 2092
    361 / 2092
    362 / 2092
    363 / 2092
    364 / 2092
    365 / 2092
    366 / 2092
    367 / 2092
    368 / 2092
    369 / 2092
    370 / 2092
    371 / 2092
    372 / 2092
    373 / 2092
    374 / 2092
    375 / 2092
    376 / 2092
    377 / 2092
    378 / 2092
    379 / 2092
    380 / 2092
    381 / 2092
    382 / 2092
    383 / 2092
    384 / 2092
    385 / 2092
    386 / 2092
    387 / 2092
    388 / 2092
    389 / 2092
    390 / 2092
    391 / 2092
    392 / 2092
    393 / 2092
    394 / 2092
    395 / 2092
    396 / 2092
    397 / 2092
    398 / 2092
    399 / 2092
    400 / 2092
    401 / 2092
    402 / 2092
    403 / 2092
    404 / 2092
    405 / 2092
    406 / 2092
    407 / 2092
    408 / 2092
    409 / 2092
    410 / 2092
    411 / 2092
    412 / 2092
    413 / 2092
    414 / 2092
    415 / 2092
    416 / 2092
    417 / 2092
    418 / 2092
    419 / 2092
    420 / 2092
    421 / 2092
    422 / 2092
    423 / 2092
    424 / 2092
    425 / 2092
    426 / 2092
    427 / 2092
    428 / 2092
    429 / 2092
    430 / 2092
    431 / 2092
    432 / 2092
    433 / 2092
    434 / 2092
    435 / 2092
    436 / 2092
    437 / 2092
    438 / 2092
    439 / 2092
    440 / 2092
    441 / 2092
    442 / 2092
    443 / 2092
    444 / 2092
    445 / 2092
    446 / 2092
    447 / 2092
    448 / 2092
    449 / 2092
    450 / 2092
    451 / 2092
    452 / 2092
    453 / 2092
    454 / 2092
    455 / 2092
    456 / 2092
    457 / 2092
    458 / 2092
    459 / 2092
    460 / 2092
    461 / 2092
    462 / 2092
    463 / 2092
    464 / 2092
    465 / 2092
    466 / 2092
    467 / 2092
    468 / 2092
    469 / 2092
    470 / 2092
    471 / 2092
    472 / 2092
    473 / 2092
    474 / 2092
    475 / 2092
    476 / 2092
    477 / 2092
    478 / 2092
    479 / 2092
    480 / 2092
    481 / 2092
    482 / 2092
    483 / 2092
    484 / 2092
    485 / 2092
    486 / 2092
    487 / 2092
    488 / 2092
    489 / 2092
    490 / 2092
    491 / 2092
    492 / 2092
    493 / 2092
    494 / 2092
    495 / 2092
    496 / 2092
    497 / 2092
    498 / 2092
    499 / 2092
    500 / 2092
    501 / 2092
    502 / 2092
    503 / 2092
    504 / 2092
    505 / 2092
    506 / 2092
    507 / 2092
    508 / 2092
    509 / 2092
    510 / 2092
    511 / 2092
    512 / 2092
    513 / 2092
    514 / 2092
    515 / 2092
    516 / 2092
    517 / 2092
    518 / 2092
    519 / 2092
    520 / 2092
    521 / 2092
    522 / 2092
    523 / 2092
    524 / 2092
    525 / 2092
    526 / 2092
    527 / 2092
    528 / 2092
    529 / 2092
    530 / 2092
    531 / 2092
    532 / 2092
    533 / 2092
    534 / 2092
    535 / 2092
    536 / 2092
    537 / 2092
    538 / 2092
    539 / 2092
    540 / 2092
    541 / 2092
    542 / 2092
    543 / 2092
    544 / 2092
    545 / 2092
    546 / 2092
    547 / 2092
    548 / 2092
    549 / 2092
    550 / 2092
    551 / 2092
    552 / 2092
    553 / 2092
    554 / 2092
    555 / 2092
    556 / 2092
    557 / 2092
    558 / 2092
    559 / 2092
    560 / 2092
    561 / 2092
    562 / 2092
    563 / 2092
    564 / 2092
    565 / 2092
    566 / 2092
    567 / 2092
    568 / 2092
    569 / 2092
    570 / 2092
    571 / 2092
    572 / 2092
    573 / 2092
    574 / 2092
    575 / 2092
    576 / 2092
    577 / 2092
    578 / 2092
    579 / 2092
    580 / 2092
    581 / 2092
    582 / 2092
    583 / 2092
    584 / 2092
    585 / 2092
    586 / 2092
    587 / 2092
    588 / 2092
    589 / 2092
    590 / 2092
    591 / 2092
    592 / 2092
    593 / 2092
    594 / 2092
    595 / 2092
    596 / 2092
    597 / 2092
    598 / 2092
    599 / 2092
    600 / 2092
    601 / 2092
    602 / 2092
    603 / 2092
    604 / 2092
    605 / 2092
    606 / 2092
    607 / 2092
    608 / 2092
    609 / 2092
    610 / 2092
    611 / 2092
    612 / 2092
    613 / 2092
    614 / 2092
    615 / 2092
    616 / 2092
    617 / 2092
    618 / 2092
    619 / 2092
    620 / 2092
    621 / 2092
    622 / 2092
    623 / 2092
    624 / 2092
    625 / 2092
    626 / 2092
    627 / 2092
    628 / 2092
    629 / 2092
    630 / 2092
    631 / 2092
    632 / 2092
    633 / 2092
    634 / 2092
    635 / 2092
    636 / 2092
    637 / 2092
    638 / 2092
    639 / 2092
    640 / 2092
    641 / 2092
    642 / 2092
    643 / 2092
    644 / 2092
    645 / 2092
    646 / 2092
    647 / 2092
    648 / 2092
    649 / 2092
    650 / 2092
    651 / 2092
    652 / 2092
    653 / 2092
    654 / 2092
    655 / 2092
    656 / 2092
    657 / 2092
    658 / 2092
    659 / 2092
    660 / 2092
    661 / 2092
    662 / 2092
    663 / 2092
    664 / 2092
    665 / 2092
    666 / 2092
    667 / 2092
    668 / 2092
    669 / 2092
    670 / 2092
    671 / 2092
    672 / 2092
    673 / 2092
    674 / 2092
    675 / 2092
    676 / 2092
    677 / 2092
    678 / 2092
    679 / 2092
    680 / 2092
    681 / 2092
    682 / 2092
    683 / 2092
    684 / 2092
    685 / 2092
    686 / 2092
    687 / 2092
    688 / 2092
    689 / 2092
    690 / 2092
    691 / 2092
    692 / 2092
    693 / 2092
    694 / 2092
    695 / 2092
    696 / 2092
    697 / 2092
    698 / 2092
    699 / 2092
    700 / 2092
    701 / 2092
    702 / 2092
    703 / 2092
    704 / 2092
    705 / 2092
    706 / 2092
    707 / 2092
    708 / 2092
    709 / 2092
    710 / 2092
    711 / 2092
    712 / 2092
    713 / 2092
    714 / 2092
    715 / 2092
    716 / 2092
    717 / 2092
    718 / 2092
    719 / 2092
    720 / 2092
    721 / 2092
    722 / 2092
    723 / 2092
    724 / 2092
    725 / 2092
    726 / 2092
    727 / 2092
    728 / 2092
    729 / 2092
    730 / 2092
    731 / 2092
    732 / 2092
    733 / 2092
    734 / 2092
    735 / 2092
    736 / 2092
    737 / 2092
    738 / 2092
    739 / 2092
    740 / 2092
    741 / 2092
    742 / 2092
    743 / 2092
    744 / 2092
    745 / 2092
    746 / 2092
    747 / 2092
    748 / 2092
    749 / 2092
    750 / 2092
    751 / 2092
    752 / 2092
    753 / 2092
    754 / 2092
    755 / 2092
    756 / 2092
    757 / 2092
    758 / 2092
    759 / 2092
    760 / 2092
    761 / 2092
    762 / 2092
    763 / 2092
    764 / 2092
    765 / 2092
    766 / 2092
    767 / 2092
    768 / 2092
    769 / 2092
:::
:::

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
