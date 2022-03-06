# Geometric priors make dataset variations vanish
Working on the repo now. May take several days to finalize...
Official implementation: [Deep vanishing point detection: Geometric priors make dataset variations vanish](), CVPR'22 

[Yancong Lin](https://yanconglin.github.io/), [Ruben Wiersma](https://rubenwiersma.nl/), [Silvia Laura Pintea](https://silvialaurapintea.github.io/), [Klaus Hildebrandt](https://graphics.tudelft.nl/~klaus/), [Elmar Eisemann](https://graphics.tudelft.nl/~eisemann/) and [Jan C. van Gemert](http://jvgemert.github.io/)

E-mail: y.lin-1ATtudelftDOTnl; r.t.wiersmaATtudelftDOTnl

Joint work from [Computer Vision Lab](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/computer-vision-lab/) and [Computer Graphics and Visualization](https://graphics.tudelft.nl/) <br/> Delft University of Technology, the Netherlands

## Introduction

Deep learning has greatly improved vanishing point detection in images. Yet, deep networks require expensive annotated datasets trained on costly hardware and do not generalize to even slightly different domains and minor problem variants. Here, we address these issues by injecting deep vanishing point detection networks with prior knowledge. This prior knowledge no longer needs to be learned from data, saving valuable annotation efforts and compute, unlocking realistic few-sample scenarios, and reducing the impact of domain changes. Moreover, because priors are interpretable, it is easier to adapt deep networks to minor problem variations such as switching between Manhattan and non-Manhattan worlds. We incorporate two end-to-end trainable geometric priors: (i) <strong>Hough Transform</strong> -- mapping image pixels to straight lines, and (ii) <strong>Gaussian sphere</strong> -- mapping lines to great circles whose intersections denote vanishing points. Experimentally, we ablate our choices and show comparable accuracy as existing models in the large-data setting. We then validate that our model improves data efficiency, is robust to domain changes, and can easily be adapted to a non-Manhattan setting.


## Main Features: added Hough line priors

 <img src="ht-lcnn/figs/exp_gt.png" width="160">   <img src="ht-lcnn/figs/exp_pred.png" width="160">   <img src="ht-lcnn/figs/exp_input.png" width="160">   <img src="ht-lcnn/figs/exp_iht.png" width="160"> 
  
 From left to right:  Ground Truth, Predictions, Input features with noise, and HT-IHT features. 
 
 The added line priors are able to localize line cadidates from the noisy input.
 
