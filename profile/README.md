<h1 align="center">Learning and Vision Efficiency Research</h1>

A research team at [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore.
<div align="center">
  <img src="https://github.com/horseee/DeepCache/raw/master/assets/svd.gif" width="50%" ></img>
  <br>
  <em>
      (1.7x acceleration of SVD-XT) 
  </em>
</div>


## Papers
- [[CVPR'23] DepGraph: Towards Any Structural Pruning](#-depgraph-towards-any-structural-pruning)  [![Star](https://img.shields.io/github/stars/vainf/torch-pruning.svg?style=social&label=Star)](https://github.com/vainf/torch-pruning)
- [[NeurIPS'23] LLM-Pruner: On the Structural Pruning of Large Language Models](#) [![Star](https://img.shields.io/github/stars/horseee/LLM-Pruner.svg?style=social&label=Star)](https://github.com/horseee/LLM-Pruner)
- [DeepCache: Accelerating Diffusion Models for Free](#-deepcache) [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache)


### 🌟 DepGraph: Towards Any Structural Pruning  
> [![Star](https://img.shields.io/github/stars/vainf/torch-pruning.svg?style=social&label=Star)](https://github.com/vainf/torch-pruning) [![Publish](https://img.shields.io/badge/conference-CVPR'23-red)]() <a href="https://pepy.tech/project/Torch-Pruning"><img src="https://static.pepy.tech/badge/Torch-Pruning?color=2196f3" alt="Downloads"></a>  
> **TL;DR**: A *general* and *fully automatic* method, Dependency Graph (DepGraph), to explicitly model the dependency between layers and comprehensively group coupled parameters for pruning.  
> *[Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Mingli Song](https://person.zju.edu.cn/en/msong), [Michael Bi Mi](https://dblp.org/pid/317/0937.html), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*     
> [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) [[GitHub]](https://github.com/vainf/torch-pruning) 
> <details> <summary>Abstract:</summary> Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS. </details>

### 🌟 LLM-Pruner: On the Structural Pruning of Large Language Models
> [![Star](https://img.shields.io/github/stars/horseee/LLM-Pruner.svg?style=social&label=Star)](https://github.com/horseee/LLM-Pruner) [![Publish](https://img.shields.io/badge/Conference-NeurIPS'23-red)]()  
> **TL;DR**: Compress your LLMs to any size! A task-agnostic compression framework with only 3 minutes for pruning and 3 hours for post-training.  
> *[Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*
> [[Paper]](https://arxiv.org/abs/2305.11627) [[GitHub]](https://github.com/horseee/LLM-Pruner)
> <details> <summary>Abstract:</summary> Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-Pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality. To this end, the performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data. We validate the LLM-Pruner on three LLMs, including LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still exhibit satisfactory capabilities in zero-shot classification and generation. </details>

### 🌟 Diff-Pruning
A differential approach to model pruning.
- **GitHub**: [Diff-Pruning](https://github.com/VainF/Diff-Pruning)

### 🌟 DeepCache: Accelerating Diffusion Models for Free
> [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache) <a href="https://pepy.tech/project/DeepCache"><img src="https://static.pepy.tech/badge/DeepCache?color=2196f3" alt="Downloads"></a>    
> **TL;DR**: A training-free paradigm that accelerates diffusion models. 2.3× speedup for Stable Diffusion v1.5 and a 4.1× speedup for LDM-4-G, based upon DDIM/PLMS.  
> *[Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*  
> [[Paper]](https://arxiv.org/abs/2312.00858) [[GitHub]](https://github.com/horseee/DeepCache)  [[Project Page]](https://horseee.github.io/Diffusion_DeepCache/) 
> <details> <summary>Abstract:</summary> Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS. </details>


### 🌟 SlimSAM
Streamlined model optimization for SAM architectures.
- **GitHub**: [SlimSAM](https://github.com/czg1225/SlimSAM)






