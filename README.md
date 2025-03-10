# Prompt Learning via Meta-Regularization


Official implementation of CVPR 2024 paper "[Prompt Learning via Meta-Regularization](https://arxiv.org/pdf/2404.00851)".
> Jinyoung Park, Juyeon Ko, Hyunwoo J. Kim. 
>
> Department of Computer Science and Engineering, Korea University

![main figure](docs/prometar.png)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-stanford-cars-1)](https://paperswithcode.com/sota/prompt-engineering-on-stanford-cars-1?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-ucf101)](https://paperswithcode.com/sota/prompt-engineering-on-ucf101?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-dtd)](https://paperswithcode.com/sota/prompt-engineering-on-dtd?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-eurosat)](https://paperswithcode.com/sota/prompt-engineering-on-eurosat?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-fgvc-aircraft)](https://paperswithcode.com/sota/prompt-engineering-on-fgvc-aircraft?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-oxford-102-flower)](https://paperswithcode.com/sota/prompt-engineering-on-oxford-102-flower?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-food-101)](https://paperswithcode.com/sota/prompt-engineering-on-food-101?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-sun397)](https://paperswithcode.com/sota/prompt-engineering-on-sun397?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-caltech-101)](https://paperswithcode.com/sota/prompt-engineering-on-caltech-101?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-imagenet)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet?p=prompt-learning-via-meta-regularization)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-learning-via-meta-regularization/prompt-engineering-on-oxford-iiit-pet-dataset)](https://paperswithcode.com/sota/prompt-engineering-on-oxford-iiit-pet-dataset?p=prompt-learning-via-meta-regularization)

## Installation
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md)

## Data Preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

<!-- # Training  -->

## ProMetaR Training

We provide bash scripts in [scripts/](../scripts) for training ProMetaR.
Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `ProMetaR/`.
Below we provide training and testing instructions for ProMetaR.

### ProMetaR

#### (1) Base-to-Novel class generalization setting
The base-to-novel ProMetaR configuration is provided in config file at `configs/trainers/ProMetaR/vit_b16_c2_ep10_batch4_4+4ctx.yaml`. All hyper-parameters such as learning rate, number of epochs, prompt length and prompt depth etc., can be modified using this config file.

Run the commands below to train ProMetaR on eurosat.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/prometar/base2new_train.sh eurosat 1 0
# evaluates on novel classes
bash scripts/prometar/base2new_test.sh eurosat 1 0

# seed=2
# trains and evaluates on base classes
bash scripts/prometar/base2new_train.sh eurosat 2 0
# evaluates on novel classes
bash scripts/prometar/base2new_test.sh eurosat 2 0

# seed=3
# trains and evaluates on base classes
bash scripts/prometar/base2new_train.sh eurosat 3 0
# evaluates on novel classes
bash scripts/prometar/base2new_test.sh eurosat 3 0
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– eurosat/
|   |   |   |–– shots_16/
|   |   |   |   |–– ProMetaR/
|   |   |   |   |   |–– vit_b16_c2_ep10_batch4_4+4ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– eurosat/
|   |   |   |–– shots_16/
|   |   |   |   |–– ProMetaR/
|   |   |   |   |   |–– vit_b16_c2_ep10_batch4_4+4ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python output/base2new/train_base/eurosat/shots_16/ProMetaR/vit_b16_c2_ep10_batch4_4+4ctx --test-log
# averaged results for novel classes
python output/base2new/test_new/eurosat/shots_16/ProMetaR/vit_b16_c2_ep10_batch4_4+4ctx --test-log
```

The above steps can be repeated for other individual datasets.



<hr />

## Citation
If you find our work, or this repository useful, please consider giving a star :star: and citation.
```bibtex
@InProceedings{Park_2024_CVPR,
    author    = {Park, Jinyoung and Ko, Juyeon and Kim, Hyunwoo J.},
    title     = {Prompt Learning via Meta-Regularization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at lpmn678@korea.ac.kr.


## Acknowledgements

Our code is based on [PromptSRC](https://github.com/muzairkhattak/PromptSRC), along with [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.


