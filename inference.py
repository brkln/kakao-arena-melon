# -*- coding: utf-8 -*-
import fire
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
import random
import implicit
from arena_util import write_json
from arena_util import remove_seen

random.seed(0)

class Infer:
    def _generate_answers(self, val):
        # 0
        popular_num = 615142
        trial = 36
        # 1
        with open("popular_song_dict.pkl", "rb") as f:
            popular_song_dict = pickle.load(f)
        with open("popular_tag_dict.pkl", "rb") as f:
            popular_tag_dict = pickle.load(f)
        with open("trainval_id_dict.pkl", "rb") as f:
            trainval_id_dict = pickle.load(f)
        songtag_matrix = sparse.load_npz('songtag_matrix_{}.npz'.format(trial))
        with open('model_{}.sav'.format(trial), 'rb') as f:
            model = pickle.load(f)

        print("done 1")

        song_fill = []
        tag_fill = []
        for j in [trainval_id_dict[i] for i in val.id.values]:
            song_fill.append([popular_song_dict[k] for k,_ in model.recommend(j, songtag_matrix, filter_items = range(popular_num, songtag_matrix.shape[1]), N = 200)])
            tag_fill.append([popular_tag_dict[k] for k in [i for i,_ in model.rank_items(j, songtag_matrix, list(range(popular_num, songtag_matrix.shape[1])))[:15]]])

        print("done 2")
    
        answers = []
        for i in range(len(val)):
            answers.append({
                "id": val.id.values[i],
                "songs": remove_seen(val.songs.values[i], song_fill[i])[:100],
                "tags": remove_seen(val.tags.values[i], tag_fill[i])[:10]
            })

        print("done 3")

        return answers

    def run(self, question_fname):
        print("Loading question file...")
        questions = pd.read_json(question_fname)

        print("Writing answers...")
        answers = self._generate_answers(questions)
        write_json(answers, "results/results.json")

if __name__ == "__main__":
    fire.Fire(Infer)