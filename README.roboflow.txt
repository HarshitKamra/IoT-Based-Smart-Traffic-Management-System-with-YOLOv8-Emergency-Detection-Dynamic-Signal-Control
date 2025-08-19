
Priority Vehicle - v4 2025-06-24 10:13pm
==============================

This dataset was exported via roboflow.com on June 29, 2025 at 8:55 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 3306 images.
Vehicle are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* Random brigthness adjustment of between -30 and 0 percent
* Random Gaussian blur of between 0 and 1.4 pixels
* Salt and pepper noise was applied to 1.88 percent of pixels


