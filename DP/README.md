# Dialogue Policy

### 1. Model checkpoints of the full training experiments:

|               | Precision        |Recall |F1-score|
|:-------------:|:-------------:|:-----:|:-----:|
| PPTOD-small |56.27% |53.46%|54.83%|


### 2. Training
To train a new model, you can use the provided scripts.

```yaml
cd ./sh_folder/X/training/ 
chmod +x ./pptod_X_full_training.sh
./pptod_X_full_training.sh
```
Here, X is in ['small', 'base', 'large'] and some key parameters are described below:

```yaml            
--train_data_ratio: The portion of training data used, default value is 1.0, meaning 100% of training data.
                    For different few-shot settings, you can set this argument to different values. For 
                    example, when train_data_ratio equals to 0.01, the model is trained with 1% of training data.
                    
--gradient_accumulation_steps: How many forward computations between two gradient updates.

--number_of_gpu: Number of avaliable GPUs.

--batch_size_per_gpu: The batch size for each GPU.
```

**[Note 1]** The few-shot training samples are randomly selected, thus the results from different runs may not be the same.

**[Note 2]** The actual batch size equals to gradient_accumulation_steps x number_of_gpu x batch_size_per_gpu. We 
recommend the actual batch size value is set as 128.


