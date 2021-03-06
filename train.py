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


    def get_token(self, title, tokenizer):        # "get_token" function source: https://arena.kakao.com/forum/topics/226

        if len(title)== 0 or title== ' ' or title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000':
            return []

        result = tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]
        return result


    def _train(self, train_json, train, val):

        # Setting the number of popular songs and tags for user-item matrix
        # Chose to use the whole songs and tags since such combination showed the best performance
        total_song = []
        for l in train.songs:
            total_song.extend(l)
        total_song = set(total_song)

        total_tag = []
        for l in train.tags:
            total_tag.extend(l)
        total_tag = set(total_tag)

        songtag_length = [len(total_song), len(total_tag)]
        with open("songtag_length.pkl", "wb") as f:
            pickle.dump(songtag_length, f)

        total_num = max(total_song) + 1
        popular_num_song = songtag_length[0]
        popular_num_tag = songtag_length[1]

        # Collecting all the unique tags in the train data
        train_tag = []
        for l in train.tags:
            train_tag.extend(l)

        train_tag_unique = sorted(set(train_tag))

        # Creating a dictionary for allocating an integer for each tag
        train_tag_dict = {}
        for i,j in enumerate(train_tag_unique):
            train_tag_dict[j] = i + total_num

        # Creating dictionaries for interpreting the result
        _, popular_song = most_popular(train_json, 'songs', popular_num_song)
        _, popular_tag_str = most_popular(train_json, 'tags', popular_num_tag)
        popular_tag = [train_tag_dict[i] for i in sorted(popular_tag_str)]

        popular_song_dict = {}
        for i,j in enumerate(sorted(popular_song)):
            popular_song_dict[i] = j
        
        with open("popular_song_dict.pkl", "wb") as f:
            pickle.dump(popular_song_dict, f)

        popular_tag_dict = {}
        for i,j in enumerate(sorted(popular_tag_str)):
            popular_tag_dict[i + popular_num_song] = j
        
        with open("popular_tag_dict.pkl", "wb") as f:
            pickle.dump(popular_tag_dict, f)


        ###########################
        print("Finished 1st step!")
        ###########################


        # Creating a new column for (songs + integer-ized tags) in the train data
        train['tag_to_num'] = train['tags'].map(lambda x: [train_tag_dict[i] for i in x])
        train['songtag'] = train.songs + train.tag_to_num

        # Creating a new column for (songs + integer-ized tags) in the val/test data
        # (includes tokenization of playlist title in order to solve cold-start problem)
        tokenizer = KhaiiiApi()
        val['token'] = val['plylst_title'].map(lambda x: self.get_token(x, tokenizer))
        val['token'] = val['token'].map(lambda x: [i[0] for i in list(filter(lambda x: x[0] in train_tag_unique, x))])
        val['tags_refined'] = val.tags + val.token
        val['tags_refined'] = val['tags_refined'].map(lambda x: list(set(x)))
        val['tags_refined'] = val['tags_refined'].map(lambda x: list(filter(lambda x: x in train_tag_unique, x)))
        val['tag_to_num'] = val['tags_refined'].map(lambda x: [train_tag_dict[i] for i in x])
        val['songtag'] = val.songs + val.tag_to_num
        val = val.drop(['token', 'tags_refined'], axis = 1)

        # Concatenating train & val/test dataframes for user-item matrix
        trainval = pd.concat([train, val], ignore_index = True)

        # Creating a dictionary for matching the correct user in the reduced user-item matrix
        trainval_id_dict = {}
        for i,j in enumerate(sorted(trainval.id.values)):
            trainval_id_dict[j] = i

        with open("trainval_id_dict.pkl", "wb") as f:
            pickle.dump(trainval_id_dict, f)


        ###########################
        print("Finished 2nd step!")
        ###########################


        # Creating user-item matrix
        # All the integers in the list "data" are tuned through several trials
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
                    data.append(120)
                elif i in popular_song[50:150]:
                    data.append(130)
                elif i in popular_song[150:300]:
                    data.append(140)
                elif i in popular_song[300:500]:
                    data.append(150)
                elif i in popular_song[500:700]:
                    data.append(160)
                elif i in popular_song[700:1000]:
                    data.append(170)
                elif i in popular_song[1000:1500]:
                    data.append(180)
                elif i in popular_song[1500:2000]:
                    data.append(190)
                else:
                    data.append(200)
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
                else:
                    data.append(10)

        songtag_matrix = sparse.csr_matrix((data, (rows, cols)))
        songtag_matrix = songtag_matrix[sorted(set(trainval.id.values)), :]                 # Excluding users that are not in train or val/test data
        songtag_matrix = songtag_matrix[:, sorted(popular_song) + sorted(popular_tag)]      # Excluding non-popular songs and tags

        sparse.save_npz('songtag_matrix.npz', songtag_matrix)

        model = implicit.als.AlternatingLeastSquares()      # Recommendation model (implicit feedback collaborative filtering using ALS)
        model.fit(songtag_matrix.T)

        with open('model.sav', 'wb') as f:
            pickle.dump(model, f)


        ###########################
        print("Finished 3rd step!")
        print("Finished training!")
        ###########################


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