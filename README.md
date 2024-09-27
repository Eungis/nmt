# [🦎] CHAMELEON [🦎]

## **Project**
1. <b>Summary</b>
- Translator using parallel corpus derived from AI hub
- Try sequence-to-sequence, vanilla transformer, reinforcement learning, dual learning, LLM.

---
## **Installation & Execution**

1. <b>Install CHAMELEON package</b>
---
* use requirements.txt

  ```shell
  pip install -r requirements.txt
  ```

<br>

3. <b>Run python script</b>
---
* Inside chameleon_train.sh
  - Sequence to sequence

  <br>

    ```shell
    python3 ./chameleon_train.py \
    --model_fn <YOUR_MODEL_PATH>/chameleon.seq2seq.pth \
    --train_fn <YOUR_DATA_PATH>/chameleon.train.tok.pickle \
    --valid_fn <YOUR_DATA_PATH>/chameleon.valid.tok.pickle \
    --src_tgt koen \
    --with_text 0 \
    --gpu_id -1 \
    --batch_size 64 \
    --n_epochs 10 \
    --max_length 256 \
    --dropout .2 \
    --word_vec_size 512 \
    --hidden_size 768 \
    --n_layers 4 \
    --max_grad_norm 1e+8 \
    --iteration_per_update 2 \
    --lr 1e-3 \
    --lr_step 0 \
    --use_adam \
    --rl_n_epochs 0
    ```
  - Transformer

  <br>

    ```shell
    python3 ./chameleon_train.py \
    --model_fn <YOUR_MODEL_PATH>/chameleon.transformer.pth \
    --train_fn <YOUR_DATA_PATH>/chameleon.train.tok.pickle \
    --valid_fn <YOUR_DATA_PATH>/chameleon.valid.tok.pickle \
    --src_tgt koen \
    --with_text 0 \
    --gpu_id 0 \
    --use_mps \
    --batch_size 256 \
    --n_epochs 2 \
    --max_length 256 \
    --dropout .2 \
    --hidden_size 768 \
    --n_layers 4 \
    --max_grad_norm 1e+8 \
    --iteration_per_update 2 \
    --lr 1e-3 \
    --lr_step 0 \
    --use_adam \
    --use_transformer \
    --rl_n_epochs 0
    ```


---
## **Prerequisite**
1. <b>Dataset</b>
- Require parallel corpus data.
- Chameleon project uses corpus data downloaded from AI Hub, which consists of Korean-English pair language set.

2. <b>Memory & GPU</b>
- Require GPU to train.
- GPU Specs:
  - Memory Speed:19.5 Gbps.
  - Digital Max Resolution:7680x4320.
  - Chipset: NVIDIA GeForce RTX 3090.
  - TRI FROZR 2 Thermal Design.
  - Video Memory: 24GB GDDR6X.
  - Memory Interface: 384-bit.
  - Output: DisplayPort x 3 (v1.4a) / HDMI 2.1 x 1.
- Batch size can be up to 256.
- Around 15G if batch size equals to 128.
