# Geometric priors make dataset variations vanish

Official implementation: [Deep vanishing point detection: Geometric priors make dataset variations vanish](), CVPR'22 

[Yancong Lin](https://yanconglin.github.io/), [Ruben Wiersma](https://rubenwiersma.nl/), [Silvia Laura Pintea](https://silvialaurapintea.github.io/), [Klaus Hildebrandt](https://graphics.tudelft.nl/~klaus/), [Elmar Eisemann](https://graphics.tudelft.nl/~eisemann/) and [Jan C. van Gemert](http://jvgemert.github.io/)

E-mail: y.lin-1ATtudelftDOTnl; r.t.wiersmaATtudelftDOTnl

Joint work from [Computer Vision Lab](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/) and [Computer Graphics and Visualization](https://graphics.tudelft.nl/) <br/> Delft University of Technology, the Netherlands

<img src="figs/overview.png" width="1024"> 


## Introduction
Deep learning has greatly improved vanishing point detection in images. Yet, deep networks require expensive annotated datasets trained on costly hardware and do not generalize to even slightly different domains and minor problem variants. Here, we address these issues by injecting deep vanishing point detection networks with prior knowledge. This prior knowledge no longer needs to be learned from data, saving valuable annotation efforts and compute, unlocking realistic few-sample scenarios, and reducing the impact of domain changes. Moreover, because priors are interpretable, it is easier to adapt deep networks to minor problem variations such as switching between Manhattan and non-Manhattan worlds. We incorporate two end-to-end trainable geometric priors: (i) <strong>Hough Transform</strong> -- mapping image pixels to straight lines, and (ii) <strong>Gaussian sphere</strong> -- mapping lines to great circles whose intersections denote vanishing points. Experimentally, we ablate our choices and show comparable accuracy as existing models in the large-data setting. We then validate that our model improves data efficiency, is robust to domain changes, and can easily be adapted to a non-Manhattan setting.


 ## Main Feature: Images - Hough Transform - Gaussian Sphere
 <img src="figs/model.png" width="600"> 
 An overview of our model for vanishing point detection, with two geometric priors.
 
 
## Main Result: Manhanttan / non-Manhattan / domain-shift
 <img src="figs/scannet_100.png" width="240">   <img src="figs/nyu_auc.png" width="240">   <img src="figs/yud_100.png" width="240"> 
 
 (i) Competitive results on large-scale Manhattan datasets: SU3/ScanNet;
 
 (ii) <strong>Advantage in detecting a varying number of VPs in non-Manhattan world: NYU Depth</strong>;
 
 (iii) <strong>Excellent performance on new datasets, e.g. train on SU3 (synthetic)/ test on YUD (real-world)</strong>.
 
 
## Data-Efficiency: superiority in small-data regime.
 <img src="figs/aa10_scan_log.png" width="320">   <img src="figs/aa10_su3_log.png" width="320">  
 

## Reproducing Results
We made minor changes on top of [NeurVPS](https://github.com/zhou13/neurvps) to fit our design. Many thanks to Yichao Zhou for releasing the code!

Note: this branch only contains the **multi-scale** version of our model, which runs at **~23FPS** on a Nvidia RTX2080Ti GPU with the **Manhattan assumption** only.

### Installation
For the ease of reproducibility, you are suggested to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/distribution/) if you prefer) before executing the following commands. Our code has been tested with miniconda/3.9, CUDA/10.2 and devtoolset/6.

```bash
conda create -y -n vpd
conda activate vpd
conda env update --file environment.yml
```

### (step 1) Processing the Dataset

SU3/ScanNet: we follow [NeurVPS](https://github.com/zhou13/neurvps) to process the data. 
```bash
cd data
../misc/gdrive-download.sh 1yRwLv28ozRvjsf9wGwAqzya1xFZ5wYET su3.tar.xz
../misc/gdrive-download.sh 1y_O9PxZhJ_Ml297FgoWMBLvjC1BvTs9A scannet.tar.xz
tar xf su3.tar.xz
tar xf tmm17.tar.xz
tar xf scannet.tar.xz
rm *.tar.xz
cd ..
```


### (step 2) Compute parameterizations: Hough Transform and Gaussian Sphere 
Shortcut: You can simply download our pre-calculated parameterizations from [SURFdrive](https://surfdrive.surf.nl/files/index.php/f/10762395210), and place them inside the project folder, e.g. `project_folder/cache/inds_32768.npz`, `project_folder/parameterization/ht_128_128_184_180.npz` and `project_folder/parameterization/sphere_neighbors_184_180_32768_rearrange.npz` folder .

To comute the mapping from pixels -HT bins - Spherical points, run the following command: 
```bash
 python parameterization.py --save_dir=parameterization/ --focal_length=1.0 --rows=128 --cols=128 --num_samples=1024 --num_points=32768 # SU3 as an example
```
It takes ~4 hours to pre-calculate the HT and sphere mappings, and 2 hours to compute `cache/inds_32768` which saves the indices for efficient sampling at multiple scales.


### (step 3) Train
We conducted all experiments on either GTX 1080Ti or RTX 2080Ti GPUs. 
To train the neural network on GPU 0 (specified by `-d 0`) with the default parameters, execute
```bash
python train.py -d 0 --identifier baseline config/nyu.yaml
```


### (step 4) Test
Manhattan world (3-orthogonal VPs):
```bash
python eval.py -d 0  -o path/to/resut.npz  path/to/config.yaml  path/to/checkpoint.pth.tar
```

You can also download our checkpoints/results/logs from [SURFdrive](https://surfdrive.surf.nl/files/index.php/f/10762395210).


## Questions:
### (1) Details about focal length.
The focal length in our code is in the unit of 2/max(h, w) pixel (where h, w are image height/width). Knowing focal length is a strongh prior as one can utilize the Manhattan assumption to find orthogonal VPs in the camera space. Given a focal length, you can use [to_pixel](https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere/blob/3e8d6c9442d8366a30a09f4386b1503d9cc1781f/parameterization.py#L78) to back-project a VP on the image plane.


### (2) Details about sampling/quantization.
Quantization details in this repo (Pixels - HT -Gaussian Sphere) are:<br/>
SU3 (*Ours**): &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 128x128 - 184x180 - 32768;<br/>
SU3 (*Ours*): &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; 256x256 - 365x180 - 32768;<br/>

Tab 1&2 show that quantization at 128x128 is already sufficient for a decent result. Moreover training/inference time decreases significantly (x2), comparing to 256x256. However, quantization has always been a weakness for the classic HT/Gaussian sphere, despite of their excelllence in adding inductive knowledge.


### (3) What are the tricks for the speedup?
They are all in the dataloader, starting from [this line](https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere/blob/f2873d1f47d92b190350301ef96e0b894c606507/vpd/datasets.py#L131) and [this line](https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere/blob/f2873d1f47d92b190350301ef96e0b894c606507/vpd/datasets.py#L201). This is an extension of our work in the main branch, where we pre-sample 32768/16384 points on the hemisphere. 

### (4) Spherical mapping on-the-fly without pre-calculation .
Unfortunately, at this momment I do not have a solution. This is also a limitation of my implementation.

### (5) Code for other baselines.
[J/T-Linkage](https://github.com/fkluger/vp-linkage); [J-Linkage](https://github.com/simbaforrest/vpdetection); [Contrario-VP](https://members.loria.fr/GSimon/software/v/); [NeurVPS](https://github.com/zhou13/neurvps); [CONSAC](https://github.com/fkluger/consac); [VaPiD?](); [Haoang Li](https://sites.google.com/view/haoangli/homepage);

### Citation
If you find our paper useful in your research, please consider citing:
```bash
@article{lin2022vpd,
  title={Deep vanishing point detection: Geometric priors make dataset variations vanish},
  author={Lin, Yancong and Wiersma, Ruben and and Pintea, Silvia L and Hildebrandt, Klaus and Eisemann, Elmar and van Gemert, Jan C},
  booktitle={Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

