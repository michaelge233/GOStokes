# GOStokes
Code for unpublished paper *Accurate Single-Shot Full-Stokes Detection Enabled by Heterogeneous Grain Orientations in Polycrystalline Films*.

This repository contains the code for dataset collection and model training to perform full-Stokes detection using the GOStokes approach proposed by Mingwei Ge, Menaxia Liu et al. The entire dataset can be found in the Code Ocean Capsule of this work to reproduce the result demonstrated in the manuscript.

## Requirements 

<b> Deeping learning: </b>
* Python 3.8.10
* PyTorch 2.0.1+cu118
* matplotlib 3.7.1
* numpy 1.24.3 
* pandas 2.0.2
* scipy 1.10.1
* tqdm 4.65.0

<b> Dataset collection:</b>

* Vimba X
* Thorcam
* ThorlabsPM100
* thorlabs_apt
* pyvisa

Adaptions of hardware ID are required to make the code work for other systems outside of our lab.

## Instructions

<b>./CollimatedBeam </b> contains code and output of experiments on collimated beam (Fig.4 in manuscript). We demonstrated the code and output for Film1 in main folder, while that of other films are in ./CollimatedBeam/other_films

<b>./Imaging </b> contains the code of our imaging experiment (Fig. 5 in manuscript). 

Code and result for deep learning were demonstrated in Jupyter notebooks (.ipynb files). They can be excuted directly after completing the dataset. 
