# Enhancing Vision-Language Model Pre-training with Image-text Pair Pruning Based on Word Frequency

# Abstract
We propose Word-Frequency-based Image-Text Pair Pruning (WFPP), a novel data pruning method that improves the efficiency of VLMs.
Unlike MetaCLIP, our method does not need metadata for pruning, but selects text-image pairs to prune based on the content of the text. Specifically, WFPP prunes text-image pairs containing high-frequency words across the entire training dataset. The effect of WFPP is to reduce the dominance of frequent words. The result is a less skewed word-frequency distribution of the dataset, known to improve the training of word embedding models. After pre-training on the pruned subset, we fine-tuned the model on the entire dataset for one additional epoch to achieve better performance. Our experiments demonstrate that applying WFPP when training a CLIP model improves performance on a wide range of downstream tasks. WFPP also provides the advantage of speeding up pre-training by using fewer samples. Additionally, we analyze the training data before and after pruning to visualize how WFPP changes the balance of word frequencies. We hope our work encourages researchers to consider the distribution of words in the training data when pre-training VLMs, not limited to CLIP.


# Results and Pre-trained Models

We will pre-train the model based on the following code and settings [Open_clip](https://github.com/mlfoundations/open_clip)

**Zero-shot classification accuracy on ImageNet-1K.**

We pre-train models by sampling image-text pairs sorted according to Equation 2 of our paper at sampling rates ranging from 50% to 90%. The pre-training dataset is **CC12M**, and the image encoder used is ViT-B-16. “Samples seen” refers to the proportion of the dataset processed during pre-training, with 100% set as 1.00. The threshold value \( t \) in Equation~\ref{equ:sub} is set to \( 10^{-7} \). “w/o FT” and “w/ FT” indicate results without and with fine-tuning, respectively.

| Method | Sample Size    | w/o FT | w/ FT | Samples seen |
|--------|----------------|--------|-------|--------------|
| CLIP   | 9.30M          | 34.8   | ✗     | 1.00×        |
| CLIP   | 4.65M (50%)    | 28.2   | 30.2  | 0.53×        |
| WFPP   | 4.65M (50%)    | 29.8   | 31.3  | 0.53×        |
| WFPP   | 5.58M (60%)    | 32.3   | 33.3  | 0.63×        |
| WFPP   | 6.51M (70%)    | 33.4   | 34.4  | 0.73×        |
| WFPP   | 7.44M (80%)    | 34.3   | 35.0  | 0.83×        |
| WFPP   | 8.37M (90%)    | **34.9** | **35.5** | 0.93×        |


# Word-Frequency-based Image-Text Pair Pruning
First, generate a word frequency dictionary using the script *data_counter.py*. Then, sort the text using *sorting_data.py* to create a list of the sorted text.
And select a specific number of samples to pre-train the model.

# Pre-training

Follow the instruction of [OpenCLIP](https://github.com/mlfoundations/open_clip) to pre-train the model.

Pre-training the model on subset or full set by

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    -m training.main \
    --train-data '/data/cc12m/cc12m-train-subset.csv' \
    --train-num-samples 5484269 \
    --model=ViT-B-16 \
    --batch-size 160 \
    --lr 1e-3 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

# Fine-tuning

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    -m training.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --model=ViT-B-16 \
    --pretrained /path/to/checkpoints/epoch_K.pt
    --batch-size 160 \
    --lr 1e-5 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

# Evaluation

We use [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark/tree/main) to evaluate CLIP and WFPP on a standard set of datasets on different tasks.
