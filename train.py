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
        total_song = []
        for l in train.songs:
            total_song.extend(l)
        total_song = set(total_song)

        total_num = 707989
        popular_num_song = len(total_song)
        popular_num_tag = 10000
        trial = 40
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

        popular_song_dict = {}
        for i,j in enumerate(sorted(popular_song)):
            popular_song_dict[i] = j
        
        with open("popular_song_dict.pkl", "wb") as f:
            pickle.dump(popular_song_dict, f)

        print("done 1")

        tokenizer = KhaiiiApi()
        train['token'] = train['plylst_title'].map(lambda x: self.get_token(x, tokenizer))
        train['token'] = train['token'].map(lambda x: [i[0] for i in list(filter(lambda x: x[0] in train_tag_unique, x))])
        train['tags_refined'] = train.tags + train.token
        train['tag_to_num'] = train['tags_refined'].map(lambda x: [train_tag_dict[i] for i in x])
        train['songtag'] = train.songs + train.tag_to_num
        # train['tag_to_num'] = train['tags'].map(lambda x: [train_tag_dict[i] for i in x])
        # train['songtag'] = train.songs + train.tag_to_num

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
                if i in popular_song[:50]:
                    data.append(10)
                elif i in popular_song[50:150]:
                    data.append(20)
                elif i in popular_song[150:300]:
                    data.append(30)
                elif i in popular_song[300:500]:
                    data.append(40)
                elif i in popular_song[500:700]:
                    data.append(50)
                elif i in popular_song[700:1000]:
                    data.append(60)
                elif i in popular_song[1000:1500]:
                    data.append(70)
                elif i in popular_song[1500:2000]:
                    data.append(80)
                # elif i in popular_song[80:450]:
                #     data.append(90)
                # elif i in popular_song[90:550]:
                #     data.append(100)
                # elif i in popular_song[100:660]:
                #     data.append(110)
                # elif i in popular_song[110:780]:
                #     data.append(100)
                # elif i in popular_song[120:130]:
                #     data.append(90)
                # elif i in popular_song[130:140]:
                #     data.append(80)
                # elif i in popular_song[140:150]:
                #     data.append(70)
                # elif i in popular_song[150:160]:
                #     data.append(60)
                # elif i in popular_song[160:170]:
                #     data.append(50)
                # elif i in popular_song[170:180]:
                #     data.append(40)
                # elif i in popular_song[180:190]:
                #     data.append(30)
                # elif i in popular_song[190:200]:
                #     data.append(20)
                else:
                    data.append(90)
            else:
                if i in popular_tag[:50]:
                    data.append(100)
                elif i in popular_tag[100:150]:
                    data.append(90)
                elif i in popular_tag[150:300]:
                    data.append(80)
                elif i in popular_tag[300:500]:
                    data.append(70)
                elif i in popular_tag[500:700]:
                    data.append(60)
                elif i in popular_tag[700:1000]:
                    data.append(50)
                elif i in popular_tag[1000:1500]:
                    data.append(40)
                elif i in popular_tag[1500:2000]:
                    data.append(30)
                elif i in popular_tag[2000:2500]:
                    data.append(20)
                # elif i in popular_tag[90:100]:
                #     data.append(20)
                else:
                    data.append(10)

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