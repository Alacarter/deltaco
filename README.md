# railrl-ut

## Introduction
This codebase contains the training code and algorithm for the DeL-TaCo paper:

**Using Both Demonstrations and Language Instructions to Efficiently Learn Robotic Tasks** <br />
Albert Yu, Raymond J. Mooney <br />
ICLR (International Conference on Learning Representations), 2023 <br />
[Web](https://deltaco-robot.github.io/) | [PDF](https://arxiv.org/pdf/2210.04476.pdf) <br />


This codebase builds on the functions/classes from the previously released repo, [cog](https://github.com/avisingh599/cog), which was released with the [COG paper](https://arxiv.org/abs/2010.14500).

## Setting up
After cloning this repo and `cd`-ing into it:
```
conda env create -f env.yml
pip install -e .
python setup.py develop
cp rlkit/launchers/config_template.py rlkit/launchers/config.py
```

Modify the `LOCAL_LOG_DIR` in `rlkit/launchers/config.py` to a path on your machine where the experiment logs will be saved.

Clone the environment [roboverse-deltaco repo](https://github.com/Alacarter/roboverse-deltaco#setup).
```
cd ..
git clone git@github.com:Alacarter/roboverse-deltaco.git
cd roboverse-deltaco
pip install -r requirements.txt
pip install -e .
```

Make sure to follow the steps to clone [bullet-objects](https://github.com/Alacarter/roboverse-deltaco#clone-bullet-objects).

Clone the [sentence transformers repo](https://github.com/UKPLab/sentence-transformers).
```
cd ..
git clone git@github.com:UKPLab/sentence-transformers.git
pip install -e .
```

### Extra setup steps
If you wish to run experiments involving CLIP as language or demo encoder, you will need to install [open_clip](https://github.com/mlfoundations/open_clip) and add the following line to your `~/.bashrc` file:

`export PYTHONPATH="$PYTHONPATH:[path_to_openclip_repo]/src"`

## Downloading Datasets
We release all our datasets, which can be downloaded [here](https://deltaco-robot.github.io/datasets). In the following commands, we use [T198_path], [T48_path], [T54_path] to denote the paths where T198, T48, and T54 dataset files were downloaded to. A similar convention is used for the eval dataset (E48+E54).


## Commands for Replicating Results
### Table 1: Scenario A--Novel Objects, Colors, and Shapes
#### Onehot upper and lower bounds
Onehot
```
python experiments/multitask_bc.py --buffers [T198_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding onehot --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1110
```

Onehot Oracle
```
python experiments/multitask_bc.py --train-target-buffers [T48+T54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding onehot --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-target-demos-per-task 400  --gpu 0 --seed 1120
```

#### Pretrained CLIP as Demo and Language Encoder

Language-only
```
python experiments/multitask_bc.py --buffers [T198_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding lang --batch-size 64 --meta-batch-size 16 --freeze-clip --clip-ckpt=[pretrained_CLIP_ckpt_path] --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1210
```

Demo-only
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals  24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299  --task-embedding demo --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 0.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 1.0 --use-cached-embs --vid-enc-cnn-type clip --freeze-clip --clip-ckpt=[pretrained_CLIP_ckpt_path]  --vid-enc-mosaic-rc=2,2 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1220
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299  --task-embedding demo_lang --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 0.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 1.0 --use-cached-embs --vid-enc-cnn-type clip --freeze-clip --clip-ckpt=[pretrained_CLIP_ckpt_path]  --vid-enc-mosaic-rc=2,2 --task-emb-input-mode film_lang_concat_video --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1230
```

#### CNN as Demo Encoder, miniLM as Language Encoder

Language-only
```
python experiments/multitask_bc.py --buffers [T198_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding lang --lang-emb-model-type minilm --latent-out-dim-size 384 --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1310
```

Demo-only
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1320
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1330
```

BC-Z (Jang et al, 2021) (Demo-only)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cosine_dist --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1340
```

MCIL (Lynch & Sermanet, 2021) (Demo-only + Language-only)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding mcil --lang-emb-model-type minilm --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --transform-targets --latent-out-dim-size 384 --task-encoder-weight 0.0 --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 1350
```

### Table 2: Scenario B--Novel Colors and Shapes

#### Onehot upper and lower bounds
Onehot
```
python experiments/multitask_bc.py --buffers [T198+T48_path] --eval-task-idx-intervals 32-35 45-49 82-85 95-99 132-135 145-149 182-185 195-199 232-235 245-249 282-285 295-299 --focus-train-task-idx-intervals 36-44 86-94 136-144 186-194 236-244 286-294 --focus-train-tasks-sample-prob 0.5 --task-embedding onehot --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0  --num-train-demos-per-task 200 --gpu 0 --seed 2110
```

Onehot Oracle
```
python experiments/multitask_bc.py --train-target-buffers [T54_path] --eval-task-idx-intervals 32-35 45-49 82-85 95-99 132-135 145-149 182-185 195-199 232-235 245-249 282-285 295-299 --task-embedding onehot --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0  --num-train-target-demos-per-task 750  --gpu 0 --seed 2120
```

#### CNN as Demo Encoder, miniLM as Language Encoder

Language-only
```
python experiments/multitask_bc.py --buffers [T198+T48_path] --eval-task-idx-intervals 32-35 45-49 82-85 95-99 132-135 145-149 182-185 195-199 232-235 245-249 282-285 295-299 --focus-train-task-idx-intervals 36-44 86-94 136-144 186-194 236-244 286-294 --focus-train-tasks-sample-prob 0.5 --task-embedding lang --lang-emb-model-type minilm --latent-out-dim-size 384 --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 2210
```

Demo-only
```
python experiments/multitask_bc.py --buffers [T198+T48_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 32-35 45-49 82-85 95-99 132-135 145-149 182-185 195-199 232-235 245-249 282-285 295-299 --focus-train-task-idx-intervals 36-44 86-94 136-144 186-194 236-244 286-294 --focus-train-tasks-sample-prob 0.5 --task-embedding demo --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 2220
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198+T48_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 32-35 45-49 82-85 95-99 132-135 145-149 182-185 195-199 232-235 245-249 282-285 295-299 --focus-train-task-idx-intervals 36-44 86-94 136-144 186-194 236-244 286-294 --focus-train-tasks-sample-prob 0.5 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 2230
```

### Table 3: How many Demonstrations is Language Worth?
The following command require running Table 1 (see the experiment with seed 1320) (CNN as Demo Encoder, miniLM as Language Encoder, Demo-only) to get a checkpoint to finetune off of.

Let `xx` $=$ number of demos per test-task that we fine tune our demo-only policy on.

Then the command is:
```
python experiments/multitask_bc.py --train-target-buffers [E48+E54_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --policy-ckpt=[params.pt path of demo-only policy] --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-target-demos-per-task xx --gpu 0 --seed 3110
```

### Table 8: Ablations
#### Language Encoder Ablations
##### DistilBERT
Language-only
```
python experiments/multitask_bc.py --buffers [T198_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding lang --lang-emb-model-type distilbert --latent-out-dim-size 768 --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8110
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type distilbert --latent-out-dim-size 768 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8120
```

##### DistilRoBERTa
Language-only
```
python experiments/multitask_bc.py --buffers [T198_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding lang --lang-emb-model-type distilroberta --latent-out-dim-size 768 --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8130
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type distilroberta --latent-out-dim-size 768 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8140
```

##### CLIP with MLP head
Language-only
```
python experiments/multitask_bc.py --buffers [T198_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding lang --batch-size 64 --meta-batch-size 12 --transform-targets --latent-out-dim-size 512 --freeze-clip --clip-ckpt=[pretrained_CLIP_ckpt_path] --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8150
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299  --task-embedding demo_lang --video-batch-size 16 --batch-size 64 --meta-batch-size 12 --transform-targets --latent-out-dim-size 512 --task-encoder-weight 0.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 1.0 --use-cached-embs --vid-enc-cnn-type clip --freeze-clip --clip-ckpt=[pretrained_CLIP_ckpt_path]  --vid-enc-mosaic-rc=2,2 --task-emb-input-mode film_lang_concat_video --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8160
```

#### Demo Encoder Ablations
##### R3M with MLP head
DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type r3m --vid-enc-mosaic-rc=2,2 --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8210
```

#### Task Encoder Loss Ablations for DeL-TaCo
Cosine distance Loss
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cosine_dist --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8310
```

No Task Encoder Loss
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 0.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8320
```

#### Task Conditioning Architecture Ablations for DeL-TaCo
Concatenating Demo + Lang to CNN Image observation embeddings
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode concat_to_img_embs --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8410
```

Concatenating Demo + Lang for FiLM layer Input
```
python experiments/multitask_bc.py --buffers [T198_path] --target-buffers [E48+E54_path] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 8420
```

### Table 9: Ambiguity Experiments (Ambiguity Scheme (ii))
Note: We are still in the process of releasing the dataset for these ambiguity experiments.

Language-only
```
python experiments/multitask_bc.py --buffers [ambigT198] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding lang --lang-emb-model-type minilm --latent-out-dim-size 384 --batch-size 64 --meta-batch-size 16 --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRAmbigObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 9110
```

Demo-only
```
python experiments/multitask_bc.py --buffers [ambigT198] --target-buffers [ambigE48+E54] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRAmbigObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 9120
```

DeL-TaCo (ours)
```
python experiments/multitask_bc.py --buffers [ambigT198] --target-buffers [ambigE48+E54] --eval-task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --task-embedding demo_lang --lang-emb-model-type minilm --latent-out-dim-size 384 --video-batch-size 16 --batch-size 64 --meta-batch-size 16 --task-encoder-weight 10.0 --task-encoder-loss-type cross_ent --task-encoder-contr-temp 0.1 --vid-enc-cnn-type plain --task-emb-input-mode film_video_concat_lang --policy-num-film-inputs 1 --env Widow250PickPlaceGRFBLRAmbigObjCSRndDistractorRndTrayQuad-v0 --num-train-demos-per-task 200 --gpu 0 --seed 9130
```

### Citation
```
@inproceedings{yu:2023,
  title={Using Both Demonstrations and Language Instructions to Efficiently Learn Robotic Tasks},
  author={Albert Yu and Raymond J. Mooney},
  booktitle={Proceedings of the International Conference on Learning Representations, 2023},
  year={2023},
}
```
