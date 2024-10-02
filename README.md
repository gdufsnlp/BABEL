# BABEL - Implementation
## Dataset
We evaluate the effectiveness of BABEL using three vulnerability datasets adopted from these papers:
- Devign [1]:https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF
- Reveal [2]:https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOyF
- FUNDED [3]:https://drive.google.com/drive/folders/1WFFV8uGi8oXpzYORyiqRCYyqJGiHSbZL?usp=sharing&pli=1

## Requirement
The major libraries are listed as follows:
- Python >= 3.8
- Torch 1.8.0
- Numpy 1.24.4
- Transformer 4.33.2
- Tokenizer 0.13.3
- tqdm

## Preprocess code
We recommend using a code formatting tool to keep each function normalized after processing, which is suitable for our subsequent extraction of inter-line dependencies based on heuristic rules.

It is important to note that we want to keep as many statements as possible in each line of the formatted code to identify the most important lines in the code, which is crucial for the vulnerability detection task. For example, when formatting C programs, we use the **Clang-format** (https://clang.llvm.org/), and we modify the `ColumnLimit` parameter of the **Clang-format** tool configuration to be as large as possible.

## Running the model
```
python run.py \
    --output_dir=./saved_models \
    --max_sentnum 150 \
    --max_wordnum 60 \
    --patience_step 20 \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=./dataset/codexglue/train.jsonl \
    --eval_data_file=./dataset/codexglue/valid.jsonl \
    --test_data_file=./dataset/codexglue/test.jsonl \
    --epoch 100 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee -a train.log
```
## Running across languages 
```
python run.py \
    --output_dir=./saved_models/model.bin \
    --max_sentnum 150 \
    --max_wordnum 60 \
    --patience_step 20 \
    --do_test \
    --train_data_file=./dataset/codexglue/train.jsonl \
    --eval_data_file=./dataset/codexglue/valid.jsonl \
    --test_data_file=./dataset/funded/JAVA/CWE-074/all.jsonl \
    --epoch 100 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 5e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee -a JAVA_CWE74.log
```
## Citation
[1]Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.

[2]Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3]Wang, H., Ye, G., Tang, Z., Tan, S. H., Huang, S., Fang, D., ... & Wang, Z. (2020). Combining graph-based learning with automated data collection for code vulnerability detection. IEEE Transactions on Information Forensics and Security, 16, 1943-1958.

[4]Lu, S., Guo, D., Ren, S., Huang, J., Svyatkovskiy, A., Blanco, A., ... & Liu, S. (2021). Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664.
