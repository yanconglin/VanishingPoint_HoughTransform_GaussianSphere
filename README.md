# Geometric priors make dataset variations vanish
Working on the repo. May take several days to finalize...

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

### Installation
For the ease of reproducibility, you are suggested to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (or [anaconda](https://www.anaconda.com/distribution/) if you prefer) before executing the following commands. Our code has been tested with CUDA-10.2 and devtoolset-6.

```bash
conda create -y -n vpd
conda activate vpd
conda env update --file environment.yml
```


### (step 1) Processing the Dataset

SU3/ScanNet: we follow [NeurVPS](https://github.com/zhou13/neurvps) to download the data. 
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

NYU/YUD: we follow [CONSAC](https://github.com/fkluger/nyu_vp) to download the data; and then process the data. 
```bash
python nyu_data_process.py
```


### (step 2) Compute parameterizations: Hough Transform and Gaussian Sphere 
Compute the mapping from pixels -HT bins - Spherical points.
We use GPUs (Pytorch) to speed up the calculation.
```bash
python parameterization_gpu.py
```
You can also download our pre-calculated parameterizations from [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/nKOCFAgZxulxHH0).

### (step 3) Training
We conducted all experiments on either GTX 1080Ti or RTX 2080Ti GPUs. 
To train the neural network on GPU 0 (specified by `-d 0`) with the default parameters, execute
```bash
python train.py -d 0 --identifier baseline config/nyu.yaml
```


### (step 3) Test
Manhattan world (3-orthogonal VPs):
```bash
python eval_manhattan.py -d 0  -o path/to/resut.npz  path/to/dataset.yaml  path/to/checkpoint.pth.tar
```

Non-Manhattan world (unknown number of VPs, one extra step - use DBSCAN to cluster VPs on the hemisphere):
```bash
python eval_nyu.py -d 0  --dump path/to/result_folder  config/nyu.yaml  path/to/checkpoint.pth.tar
python cluster_nyu.py
```

You can also download our pre-trained models from [SURFdrive](https://surfdrive.surf.nl/files/index.php/s/nKOCFAgZxulxHH0).


### ToDo: VP detection for Your Own Images


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

