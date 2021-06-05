# FaceSwapping

Digital image processing final project.

## Installation

```
$ git clone https://github.com/jay-in-git/FaceSwapping.git
$ cd FaceSwapping
$ python3 -m pip install -r requirements.txt --user
```

## Usage

```
Usage: main.py [-h] [-o OUTPUT] [--method {poisson,direct,multi}] [--option {face,head,eye,mouth,nose}] src_image tgt_image

positional arguments:
  src_image             Image with the face to be pasted
  tgt_image             Imgae with the face to be replaced

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the result image
  --method {poisson,direct,multi}
                        Blending method
  --option {face,head,eye,mouth,nose}
                        Specify specific part of face to be replaced 
```
