<h1 align="center">Learning and Vision Efficiency Research</h1>

A research team at [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore.

## Papers
- [[CVPR'23] DepGraph: Towards Any Structural Pruning](#-depgraph-towards-any-structural-pruning)  [![Star](https://img.shields.io/github/stars/vainf/torch-pruning.svg?style=social&label=Star)](https://github.com/vainf/torch-pruning)
- [DeepCache: Accelerating Diffusion Models for Free](#-deepcache) [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache)


## 🌟 DepGraph: Towards Any Structural Pruning  
> [![Star](https://img.shields.io/github/stars/vainf/torch-pruning.svg?style=social&label=Star)](https://github.com/vainf/torch-pruning) [![Publish](https://img.shields.io/badge/conference-CVPR'23-blue)]() <a href="https://pepy.tech/project/Torch-Pruning"><img src="https://static.pepy.tech/badge/Torch-Pruning?color=2196f3" alt="Downloads"></a>  
> DepGraph: Towards Any Structural Pruning    
> *[Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Mingli Song](https://person.zju.edu.cn/en/msong), [Michael Bi Mi](https://dblp.org/pid/317/0937.html), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*     
> [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) [[GitHub]](https://github.com/vainf/torch-pruning) 

<details>
<summary>Abstract:</summary>
Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS.
</details>

<div align="left">
  <img src="https://github.com/VainF/Torch-Pruning/blob/master/assets/intro.png" width="60%" ></img>
  <br>
  <em>
      (1.7x acceleration of SVD-XT) 
  </em>
</div>


## 🌟 DeepCache 
> [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache)     
> DeepCache: Accelerating Diffusion Models for Free  
> *[Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*  
> [[Paper]](https://arxiv.org/abs/2305.11627) [[GitHub]](https://github.com/horseee/DeepCache)  [[Project Page]](https://horseee.github.io/Diffusion_DeepCache/) 

<details>
<summary>Abstract:</summary>
Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS.
</details>

<div align="center">
  <img src="https://github.com/horseee/DeepCache/raw/master/assets/svd.gif" width="100%" ></img>
  <br>
  <em>
      (1.7x acceleration of SVD-XT) 
  </em>
</div>


  



## 🌟 LLM-Pruner
Efficient pruning tool for large language models.
- **GitHub**: [LLM-Pruner](https://github.com/horseee/LLM-Pruner)

## 🌟 Diff-Pruning
A differential approach to model pruning.
- **GitHub**: [Diff-Pruning](https://github.com/VainF/Diff-Pruning)


## 🌟 SlimSAM
Streamlined model optimization for SAM architectures.
- **GitHub**: [SlimSAM](https://github.com/czg1225/SlimSAM)






