# -*- coding: utf-8 -*-
import fire
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
import random
import implicit
from khaiii import KhaiiiApi
from arena_util import load_json
from arena_util import write_json
from arena_util import most_popular

random.seed(0)

class Train:
    def get_token(self, title, tokenizer):        # get_token 함수 출처: https://arena.kakao.com/forum/topics/226
        if len(title)== 0 or title== ' ' or title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000':
            return []

        result = tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]
        return result

    def _train(self, train_json, train, val):
        # 0
        total_num = 707989
        popular_num_song = 615142        # train 데이터 내 곡 개수 615142개
        popular_num_tag = 10000
        trial = 34
        # 1
        train_tag = []
        for l in train.tags:
            train_tag.extend(l)

        train_tag_unique = sorted(set(train_tag))
        # 2
        train_tag_dict = {}
        for i,j in enumerate(train_tag_unique):
            train_tag_dict[j] = i + total_num

        popular_tag_dict = {}
        for i,j in enumerate(train_tag_unique):
            popular_tag_dict[i + popular_num_song] = j
        
        with open("popular_tag_dict.pkl", "wb") as f:
            pickle.dump(popular_tag_dict, f)
        # 3
        _, popular_song = most_popular(train_json, 'songs', popular_num_song)
        _, popular_tag = most_popular(train_json, 'tags', popular_num_tag)
        popular_tag = [train_tag_dict[i] for i in popular_tag]
        song_100 = popular_song[:100]
        song_200 = popular_song[100:200]
        song_300 = popular_song[200:300]
        song_400 = popular_song[300:400]
        song_500 = popular_song[400:500]
        song_600 = popular_song[500:600]
        song_700 = popular_song[600:700]
        song_800 = popular_song[700:800]
        song_900 = popular_song[800:900]
        song_1000 = popular_song[900:1000]
        song_1100 = popular_song[1000:1100]
        song_1200 = popular_song[1100:1200]
        song_1300 = popular_song[1200:1300]
        song_1400 = popular_song[1300:1400]
        song_1500 = popular_song[1400:1500]
        song_1600 = popular_song[1500:1600]
        song_1700 = popular_song[1600:1700]
        song_1800 = popular_song[1700:1800]
        song_1900 = popular_song[1800:1900]
        song_2000 = popular_song[1900:2000]
        tag_100 = popular_tag[:100]
        tag_200 = popular_tag[100:200]
        tag_300 = popular_tag[200:300]
        tag_400 = popular_tag[300:400]
        tag_500 = popular_tag[400:500]
        tag_600 = popular_tag[500:600]
        tag_700 = popular_tag[600:700]
        tag_800 = popular_tag[700:800]
        tag_900 = popular_tag[800:900]
        tag_1000 = popular_tag[900:1000]

        popular_song_dict = {}
        for i,j in enumerate(sorted(popular_song)):
            popular_song_dict[i] = j
        
        with open("popular_song_dict.pkl", "wb") as f:
            pickle.dump(popular_song_dict, f)

        print("done 1")

        train['tag_to_num'] = train['tags'].map(lambda x: [train_tag_dict[i] for i in x])
        train['songtag'] = train.songs + train.tag_to_num

        tokenizer = KhaiiiApi()
        val['token'] = val['plylst_title'].map(lambda x: self.get_token(x, tokenizer))
        val['token'] = val['token'].map(lambda x: [i[0] for i in list(filter(lambda x: x[0] in train_tag_unique, x))])
        val['tags_refined'] = val.tags + val.token
        val['tags_refined'] = val['tags_refined'].map(lambda x: list(set(x)))
        val['tags_refined'] = val['tags_refined'].map(lambda x: list(filter(lambda x: x in train_tag_unique, x)))
        val['tag_to_num'] = val['tags_refined'].map(lambda x: [train_tag_dict[i] for i in x])
        val['songtag'] = val.songs + val.tag_to_num
        val = val.drop(['token', 'tags_refined'], axis = 1)

        trainval = pd.concat([train, val], ignore_index = True)

        trainval_id_dict = {}
        for i,j in enumerate(sorted(trainval.id.values)):
            trainval_id_dict[j] = i

        with open("trainval_id_dict.pkl", "wb") as f:
            pickle.dump(trainval_id_dict, f)

        print("done 2")

        rows = []
        for id, st in zip(trainval.id, trainval.songtag):
            rows.extend([id] * len(st))

        cols = []
        for l in trainval.songtag:
            cols.extend(l)

        data = []
        for i in cols:
            if i < total_num:
                if i in song_100:
                    data.append(10)
                elif i in song_200:
                    data.append(20)
                elif i in song_300:
                    data.append(30)
                elif i in song_400:
                    data.append(40)
                elif i in song_500:
                    data.append(50)
                elif i in song_600:
                    data.append(60)
                elif i in song_700:
                    data.append(70)
                elif i in song_800:
                    data.append(80)
                elif i in song_900:
                    data.append(90)
                elif i in song_1000:
                    data.append(100)
                elif i in song_1100:
                    data.append(110)
                elif i in song_1200:
                    data.append(120)
                elif i in song_1300:
                    data.append(130)
                elif i in song_1400:
                    data.append(140)
                elif i in song_1500:
                    data.append(150)
                elif i in song_1600:
                    data.append(160)
                elif i in song_1700:
                    data.append(170)
                elif i in song_1800:
                    data.append(180)
                elif i in song_1900:
                    data.append(190)
                elif i in song_2000:
                    data.append(200)
                else:
                    data.append(210)
            else:
                if i in tag_100:
                    data.append(10)
                elif i in tag_200:
                    data.append(20)
                elif i in tag_300:
                    data.append(30)
                elif i in tag_400:
                    data.append(40)
                elif i in tag_500:
                    data.append(50)
                elif i in tag_600:
                    data.append(60)
                elif i in tag_700:
                    data.append(70)
                elif i in tag_800:
                    data.append(80)
                elif i in tag_900:
                    data.append(90)
                elif i in tag_1000:
                    data.append(100)
                else:
                    data.append(110)

        songtag_matrix = sparse.csr_matrix((data, (rows, cols)))
        songtag_matrix = songtag_matrix[sorted(set(trainval.id.values)), :]
        songtag_matrix = songtag_matrix[:, sorted(popular_song) + list(range(total_num, songtag_matrix.shape[1]))]

        sparse.save_npz('songtag_matrix_{}.npz'.format(trial), songtag_matrix)

        model = implicit.als.AlternatingLeastSquares()
        model.fit(songtag_matrix.T)

        with open('model_{}.sav'.format(trial), 'wb') as f:
            pickle.dump(model, f)

        print("done 3")

    def run(self, train_fname, question_fname):
        print("Loading train file...")
        train_json = load_json(train_fname)
        train_data = pd.read_json(train_fname)

        print("Loading question file...")
        questions = pd.read_json(question_fname)

        print("Generating model...")
        self._train(train_json, train_data, questions)

if __name__ == "__main__":
    fire.Fire(Train)
    