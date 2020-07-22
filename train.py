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
    def get_token(self, title: str, tokenizer):     # get_token 함수 출처: https://arena.kakao.com/forum/topics/226
        if len(title)== 0 or title== ' ' or title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000':
            return []

        result = tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]
        return result

    def _train(self, train_json, train, val):
        # 0
        total_num = 707989
        # 1
        _, popular_song = most_popular(train_json, 'songs', 10000)
        # 2
        train_tag = []
        for l in train.tags:
            train_tag.extend(l)

        train_tag_unique = list(set(train_tag))
        # 3
        train_tag_dict = {}
        for i,j in enumerate(sorted(set(train_tag))):
            train_tag_dict[j] = i + total_num

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
                if i in popular_song:
                    data.append(1)
                else:
                    data.append(0)
            else:
                data.append(1)

        songtag_matrix = sparse.csr_matrix((data, (rows, cols)))
        songtag_matrix = songtag_matrix[sorted(set(trainval.id.values)), sorted(set(songtag_matrix.nonzero()[1]))]

        model = implicit.als.AlternatingLeastSquares()
        model.fit(songtag_matrix.T)

        with open('model_1.sav', 'wb') as pickle_out:
            pickle.dump(model, pickle_out)

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