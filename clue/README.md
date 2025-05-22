This repository is the official implementation of **CLUE** [Enhancing the Fine-Grained Power of Self-supervised
Learning via Deep Clustering]. 

We run our codes on 4 GTX3090 GPU.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Preparation
To run the codes, you should download the pretrained ckpt on https://download.pytorch.org/models/resnet50-11ad3fa6.pth for fair comparison.


## Training

To train the model(s) in the paper, run this command:

```train
torchrun --nproc_per_node=4 main.py --epochs 100 --warmup_epochs 10 --batch_size 128 --dump_path "path/to/your/outputdir"  --ckpt_from_impre "path/to/your/weights" --data_path "path/to/your/dataset" --is_parts pa --n_parts 4 --part_method "_global_part"  --text_path "path/to/your/json" --with_texts "sample_level"
```


## Evaluation

To evaluate my model for retrieval task on CUB200, run:

```eval
torchrun --nproc_per_node=1 eval_retrieval.py --data_path "path/to/your/dataset" --dump_path "path/to/your/outputdir" --batch_size 64 --pretrained "path/to/your/weights"
```

To evaluate my model for linear prob task on CUB200, run:
```eval
torchrun --nproc_per_node=1 eval_retrieval.py --data_path "path/to/your/dataset" --dump_path "path/to/your/outputdir" --batch_size 64 --pretrained "path/to/your/weights"
```



## Results

Our model achieves the following performance on :

### [Image Classification on CUB200]

| Model name         | Top 1 Accuracy  |
| ------------------ |---------------- | 
|LCR     |    65.24%         |
|LCR     |    66.17%         |
|CLUE    |    69.62%         |

### [Image Retrieval on CUB200]

| Model name         | Top 1   | Top 5   |
| ------------------ |---------|---------|
|LCR     |    41.26%         |  --       |
|LCR     |    42.06%         |  69.59%   |
|CLUE    |    46.83%         |  73.21%   |

