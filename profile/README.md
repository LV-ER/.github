<h1 align="center">Learning and Vision Efficiency Research</h1>

A project page for efficient deep learning at [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore.
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/example_compress.gif" width="48%" ></img>
  <img src="https://github.com/czg1225/SlimSAM/blob/master/images/paper/everything.PNG" width="48%" ></img>
  <br>
  <div align="center">
  <img src="https://github.com/horseee/DeepCache/raw/master/assets/svd.gif" width="96%" ></img>
  <br>
  <em>
      (1.7x training-free acceleration of Stable Video Diffusion-XT with DeepCache) 
  </em>
</div>
</div>


## Papers
- [[CVPR'23] DepGraph: Towards Any Structural Pruning](#-depgraph-towards-any-structural-pruning)  [![Star](https://img.shields.io/github/stars/vainf/torch-pruning.svg?style=social&label=Star)](https://github.com/vainf/torch-pruning)
- [[NeurIPS'23] LLM-Pruner: On the Structural Pruning of Large Language Models](#-llm-pruner-on-the-structural-pruning-of-large-language-models) [![Star](https://img.shields.io/github/stars/horseee/LLM-Pruner.svg?style=social&label=Star)](https://github.com/horseee/LLM-Pruner)
- [[NeurIPS'23] Structural Pruning for Diffusion Models](#-structural-pruning-for-diffusion-models) [![Star](https://img.shields.io/github/stars/VainF/Diff-Pruning.svg?style=social&label=Star)](https://github.com/VainF/Diff-Pruning)
- [DeepCache: Accelerating Diffusion Models for Free](#-deepcache-accelerating-diffusion-models-for-free) [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache)
- [SlimSAM: 0.1% Data Makes Segment Anything Slim](#-slimsam-01-data-makes-segment-anything-slim) [![Star](https://img.shields.io/github/stars/czg1225/SlimSAM.svg?style=social&label=Star)](https://github.com/czg1225/SlimSAM)
  
## 


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

### 🌟 Structural Pruning for Diffusion Models
> [![Star](https://img.shields.io/github/stars/Vainf/Diff-Pruning.svg?style=social&label=Star)](https://github.com/Vainf/Diff-Pruning) [![Publish](https://img.shields.io/badge/Conference-NeurIPS'23-red)]()    
> **TL;DR**: An efficient compression method tailored for learning lightweight diffusion models from pre-existing ones with two benefits: Efficiency and Consistency.    
> *[Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*  
> [[Paper]](https://arxiv.org/abs/2305.10924) [[GitHub]](https://github.com/vainf/Diff-Pruning)  
>  <details> <summary>Abstract:</summary> Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across several datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained models. </details>

### 🌟 DeepCache: Accelerating Diffusion Models for Free
> [![Star](https://img.shields.io/github/stars/horseee/DeepCache.svg?style=social&label=Star)](https://github.com/horseee/DeepCache) <a href="https://pepy.tech/project/DeepCache"><img src="https://static.pepy.tech/badge/DeepCache?color=2196f3" alt="Downloads"></a>    
> **TL;DR**: A training-free paradigm that accelerates diffusion models. 2.3× speedup for Stable Diffusion v1.5 and a 4.1× speedup for LDM-4-G, based upon DDIM/PLMS.   
> *[Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*  
> [[Paper]](https://arxiv.org/abs/2312.00858) [[GitHub]](https://github.com/horseee/DeepCache)  [[Project Page]](https://horseee.github.io/Diffusion_DeepCache/) 
> <details> <summary>Abstract:</summary> Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS. </details>


### 🌟 SlimSAM: 0.1% Data Makes Segment Anything Slim
> [![Star](https://img.shields.io/github/stars/czg1225/SlimSAM.svg?style=social&label=Star)](https://github.com/czg1225/SlimSAM)    
> **TL;DR**:  SlimSAM is a data-efficient SAM compression method, which offers exceptional performance with significantly low training costs (only 0.1% data).  
> *[Zigeng Chen](https://github.com/czg1225), [Gongfan Fang](https://fangggf.github.io/), [Xinyin Ma](https://horseee.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)*  
> [[Paper]](https://arxiv.org/abs/2312.05284) [[GitHub]](https://github.com/czg1225/SlimSAM)  
> <details> <summary>Abstract:</summary> The formidable model size and demanding computational requirements of Segment Anything Model (SAM) have rendered it cumbersome for deployment on resource-constrained devices. Existing approaches for SAM compression typically involve training a new network from scratch, posing a challenging trade-off between compression costs and model performance. To address this issue, this paper introduces SlimSAM, a novel SAM compression method that achieves superior performance with remarkably low training costs. This is achieved by the efficient reuse of pre-trained SAMs through a unified pruning-distillation framework. To enhance knowledge inheritance from the original SAM, we employ an innovative alternate slimming strategy that partitions the compression process into a progressive procedure. Diverging from prior pruning techniques, we meticulously prune and distill decoupled model structures in an alternating fashion. Furthermore, a novel label-free pruning criterion is also proposed to align the pruning objective with the optimization target, thereby boosting the post-distillation after pruning. SlimSAM yields significant performance improvements while demanding over 10 times less training costs than any other existing methods. Even when compared to the original SAM-H, SlimSAM achieves approaching performance while reducing parameter counts to merely 0.9% (5.7M), MACs to 0.8% (21G), and requiring only 0.1% (10k) of the SAM training data. </details>






