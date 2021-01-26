# Kakao Arena - Melon Playlist Continuation (Solution by brkln)
This repository is for Melon Playlist Continuation, the 3rd machine learning challenge of Kakao Arena.

## Environment
The codes are written based on Python 3.7.6. These are the requirements for running the codes:
- fire
- tqdm
- numpy
- pandas
- scipy
- implicit
- khaiii

## How to run
1. Before running the codes, **2 json files** should be prepared: `train.json`, `test.json`. The files can be downloaded from https://arena.kakao.com/
2. After downloading the json files, both files should be placed in a folder named `res`. The directory should look like this:
```bash
$> tree
.
├── README.md
├── arena_util.py
├── inference.py
├── requirements.txt
├── res
│   ├── test.json
│   └── train.json
└── train.py
```
3. `train.py` is run by the code below. After running the code, **6 files** will be saved in the directory: `songtag_length.pkl`, `popular_song_dict.pkl`, `popular_tag_dict.pkl`, `trainval_id_dict.pkl`, `songtag_matrix.npz`, `model.sav`.
```bash
$> python train.py run \    
        --train_fname=res/train.json \
        --question_fname=res/test.json
```
4. `inference.py` is run by the code below. After running the code, the result file will be saved in the following directory: `arena_data/results/results.json`.
```bash
$> python inference.py run \
        --question_fname=res/test.json
```

## Source
- `arena_util.py` is provided by Kakao Corp.
- Function `get_token` in `train.py` is provided in https://arena.kakao.com/forum/topics/226