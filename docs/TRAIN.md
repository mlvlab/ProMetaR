# ProMetaR Training

We provide bash scripts in [scripts/](../scripts) for training ProMetaR.
Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `ProMetaR/`.
Below we provide training and testing instructions for ProMetaR.

### Training time and compute
We train ProMetaR on each dataset with a batch size of 4 using a **single** gpu.
Training ProMetaR on ImageNet for 20 epochs takes around 8 hours for a single seed. So results for 3 seeds takes around 18 hours. For all remaining 10 datasets, it combinedly takes around around 12 hours (for all 3 seeds) on a single gpu. 

## ProMetaR

#### (1) Base-to-Novel class generalization setting
The base-to-novel ProMetaR configuration is provided in config file at `configs/trainers/ProMetaR/vit_b16_c2_ep10_batch4_4+4ctx.yaml`. All hyper-parameters such as learning rate, number of epochs, prompt length and prompt depth etc., can be modified using this config file.

Run the commands below to train ProMetaR on eurosat.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/prometar/base2new_train.sh eurosat 1
# evaluates on novel classes
bash scripts/prometar/base2new_test.sh eurosat 1

# seed=2
# trains and evaluates on base classes
bash scripts/prometar/base2new_train.sh eurosat 2
# evaluates on novel classes
bash scripts/prometar/base2new_test.sh eurosat 2

# seed=3
# trains and evaluates on base classes
bash scripts/prometar/base2new_train.sh eurosat 3
# evaluates on novel classes
bash scripts/prometar/base2new_test.sh eurosat 3
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

