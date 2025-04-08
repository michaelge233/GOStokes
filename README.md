# GOStokes
Code for unpublished paper *Accurate Single-Shot Full-Stokes Detection Enabled by Heterogeneous Grain Orientations in Polycrystalline Films*.

This repository contains the code for dataset collection and model training to perform full-Stokes detection using the GOStokes approach proposed by Mingwei Ge, Menaxia Liu et al. The dataset and trained models are available at [Zenodo](https://zenodo.org/records/15175282?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE2ZGJmMDdjLTE1NzAtNDEyMC1iMjg1LWY0ZTFmMWVjZmRiNSIsImRhdGEiOnt9LCJyYW5kb20iOiI4ZDliMGYzNjg4MDc5ZWQyYjMyMDZlY2FkYWNmM2E0MiJ9.1vZilKguiAyfgvU8j8gFxqNIpCcboX9g82uTokeZ4DsCaeo-kkvegkcZEJTkW4AEZKMxyGpC7uS4dfQovo6uNA). 

## System Requirements 

<b> Model training and test:</b>
* Python 3.8.10
* PyTorch 2.0.1+cu118
* matplotlib 3.7.1
* numpy 1.24.3 
* pandas 2.0.2
* scipy 1.10.1
* tqdm 4.65.0

All code for model training and test is in Python, no addtional compilation and installation are required.

<b> Dataset collection:</b>
* Vimba X
* Thorcam
* ThorlabsPM100
* thorlabs_apt
* pyvisa

A Thorlabs CS165MU camera and an Allied Vision Alvium 1800 U-234 were used for collimated beam and imaging experiments respectively. Adaptions are required to make the code work for other hardware systems.

## Instructions

<b>./CollimatedBeam </b> contains code and output of experiments on collimated beam (Fig.4 in manuscript). We demonstrated the code and output for Film1 in main folder, while that of other films are in ./CollimatedBeam/other_films. The function of some files are listed below:

* collect_dataset.py: Dataset collection.
* Film1.ipynb: Train and evaluate models using the whole dataset. It takes less than 10 minutes for training on a laptop with GPU.
* change_size.ipynb: Results using different training size.
* matrix_method.ipynb: Mueller matix method reference.

<b>./Imaging </b> contains the code of our imaging experiment (Fig. 5 in manuscript). The function of some files are listed below:

* measure_stokes.py: measure the training data using beams with known states of polarization and intensity profile.
* capture_one_image.py: capture the image of an object for GOStoke imaging.
* train_all_pixels.py: train the models based on training data. It should be executed on GPU clusters and takes hours to finish.
* plot.ipynb: resolve and plot Stokes parameters imagings from raw data.
* pixel_size.ipynb: experiments based on different super-pixel size.

## Reproduction Guide
1. Download the code of [this repository](https://github.com/michaelge233/GOStokes.git) and the dataset from [Zenodo](https://zenodo.org/records/15175282?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE2ZGJmMDdjLTE1NzAtNDEyMC1iMjg1LWY0ZTFmMWVjZmRiNSIsImRhdGEiOnt9LCJyYW5kb20iOiI4ZDliMGYzNjg4MDc5ZWQyYjMyMDZlY2FkYWNmM2E0MiJ9.1vZilKguiAyfgvU8j8gFxqNIpCcboX9g82uTokeZ4DsCaeo-kkvegkcZEJTkW4AEZKMxyGpC7uS4dfQovo6uNA). 
2. For collimated beam, find the data of a film. Unzip the dataset and move all folders to a same directory with the .ipynb files. For example, for film1, all folders in film1.zip should be moved to the directory containing Film1.ipynb.
3. For imaging, unzip imaging.zip and move all folders to the same directory with train_all_pixels.py and plot.ipynb.
4. Execute .py or .ipynb files.