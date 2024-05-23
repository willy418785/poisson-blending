# Poisson Image Blending

This project implements Poisson Image Editing (P. Perez et al., 2003) using the OpenCV library in Python. Poisson Image Editing is a technique used in image processing to seamlessly blend a source image region into a target image.


## Features

- Seamless cloning of image regions
- Color and gradient preservation
- Easy-to-use interface with OpenCV functions

## Requirements

- Python 3.6+
- opencv-python 4.5+
- NumPy

## Installation

Clone the repository:

```bash
git clone https://github.com/willy418785/poisson-blending
cd poisson-blending
```

## Usage

To use the Poisson Image Editing tool, run the following command:

```bash
python main.py path-to-source.jpg path-to-target.jpg
```

## Arguments

- First argument is the path to source image
- Second argument is the path to target image

## Example

- Source image: `./data/raw/sig_wood/src.jpg`
- Target image: `./data/raw/sig_wood/tar.jpg`
  
Run command below: 
```bash
python main.py ./data/raw/sig_wood/src.jpg ./data/raw/sig_wood/tar.jpg
```

This will open a window displaying sourse image. Follow intructions below to use this tool:

1. **Press left mouse button** and **drag** cursor to draw the mask for  image blending. You can use **mouse scroll** to adjust the size of the brush. Press 'S' to finish drawing.

2. **Press left mouse button** and **drag** mask to your desired location on the target image. You can use **mouse scroll** to scale the size of the mask. Press 'S' to finish mask moving.
   
3. The blending process might take a while depending on the number of pixels in mask. Blending results will be saved in the working directory named after `default.jpg`, `mix.jpg` or `average.jpg` representing results of three different types of guiding vector field selections.

## Code Overview

- MVC: This project implements MVC architecture. 

- Image data is held in `Model.Image`
  
- UI-related contents is in `View.Painter`

- Mask preparation and moving logic reside in `Controller.Masker` and `Controller.MaskMover` respectively. 
  
- The core poisson blending logic are presented in `Controller.PoissonBlender`
   
## Reference

- Pérez, P., Gangnet, M., & Blake, A. (2003). Poisson image editing. ACM Transactions on Graphics (TOG), 22(3), 313-318.