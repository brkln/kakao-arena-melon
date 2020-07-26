# -*- coding: utf-8 -*-
import fire
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
import random
import implicit
import tensorflow as tf
from khaiii import KhaiiiApi
from arena_util import load_json
from arena_util import write_json
from arena_util import most_popular
from arena_util import remove_seen

random.seed(0)

class Train:
    def get_token(self, title, tokenizer):        # get_token 함수 출처: https://arena.kakao.com/forum/topics/226
        if len(title)== 0 or title== ' ' or title == '\u3000\u3000\u3000\u3000\u3000\u3000\u3000\u3000':
            return []

        result = tokenizer.analyze(title)
        result = [(morph.lex, morph.tag) for split in result for morph in split.morphs]
        return result

    def _train(self, train_json, train, val, val_answer):
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

        w1 = tf.Variable(10., dtype=tf.float32)
        w2 = tf.Variable(20., dtype=tf.float32)
        w3 = tf.Variable(30., dtype=tf.float32)
        w4 = tf.Variable(40., dtype=tf.float32)
        w5 = tf.Variable(50., dtype=tf.float32)
        w6 = tf.Variable(60., dtype=tf.float32)
        w7 = tf.Variable(70., dtype=tf.float32)
        w8 = tf.Variable(80., dtype=tf.float32)
        w9 = tf.Variable(90., dtype=tf.float32)
        w10 = tf.Variable(100., dtype=tf.float32)
        w11 = tf.Variable(90., dtype=tf.float32)
        w12 = tf.Variable(80., dtype=tf.float32)
        w13 = tf.Variable(70., dtype=tf.float32)
        w14 = tf.Variable(60., dtype=tf.float32)
        w15 = tf.Variable(50., dtype=tf.float32)
        w16 = tf.Variable(40., dtype=tf.float32)
        w17 = tf.Variable(30., dtype=tf.float32)
        w18 = tf.Variable(20., dtype=tf.float32)
        w19 = tf.Variable(10., dtype=tf.float32)

        for epoch in range(15):

            data = []
            for i in cols:
                if i < total_num:
                    if i in popular_song[:50]:
                        data.append(w1)
                    elif i in popular_song[50:150]:
                        data.append(w2)
                    elif i in popular_song[150:300]:
                        data.append(w3)
                    elif i in popular_song[300:500]:
                        data.append(w4)
                    elif i in popular_song[500:700]:
                        data.append(w5)
                    elif i in popular_song[700:1000]:
                        data.append(w6)
                    elif i in popular_song[1000:1500]:
                        data.append(w7)
                    elif i in popular_song[1500:2000]:
                        data.append(w8)
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
                        data.append(w9)
                else:
                    if i in popular_tag[:50]:
                        data.append(w10)
                    elif i in popular_tag[100:150]:
                        data.append(w11)
                    elif i in popular_tag[150:300]:
                        data.append(w12)
                    elif i in popular_tag[300:500]:
                        data.append(w13)
                    elif i in popular_tag[500:700]:
                        data.append(w14)
                    elif i in popular_tag[700:1000]:
                        data.append(w15)
                    elif i in popular_tag[1000:1500]:
                        data.append(w16)
                    elif i in popular_tag[1500:2000]:
                        data.append(w17)
                    elif i in popular_tag[2000:2500]:
                        data.append(w18)
                    else:
                        data.append(w19)

            songtag_matrix = sparse.csr_matrix((data, (rows, cols)))
            songtag_matrix = songtag_matrix[sorted(set(trainval.id.values)), :]
            songtag_matrix = songtag_matrix[:, sorted(popular_song) + list(range(total_num, songtag_matrix.shape[1]))]

            sparse.save_npz('songtag_matrix_{}.npz'.format(trial), songtag_matrix)

            model = implicit.als.AlternatingLeastSquares()
            model.fit(songtag_matrix.T)

            with open('model_{}.sav'.format(trial), 'wb') as f:
                pickle.dump(model, f)

            print("done 3")

            song_fill = []
            tag_fill = []
            for j in [trainval_id_dict[i] for i in val.id.values]:
                song_fill.append([popular_song_dict[k] for k,_ in model.recommend(j, songtag_matrix, filter_items = range(popular_num_song, songtag_matrix.shape[1]), N = 200)])
                tag_fill.append([popular_tag_dict[k] for k in [i for i,_ in model.rank_items(j, songtag_matrix, list(range(popular_num_song, songtag_matrix.shape[1])))[:15]]])

            answers = []
            for i in range(len(val)):
                answers.append({
                    "id": val.id.values[i],
                    "songs": remove_seen(val.songs.values[i], song_fill[i])[:100],
                    "tags": remove_seen(val.tags.values[i], tag_fill[i])[:10]
                })

            result = pd.DataFrame(answers)
            result['tag_to_num'] = result['tags'].map(lambda x: [train_tag_dict[i] for i in x])
            result['songtag'] = result.songs + result.tag_to_num

            result_rows = []
            for id, st in zip(result.id, result.songtag):
                result_rows.extend([id] * len(st))

            result_cols = []
            for l in result.songtag:
                result_cols.extend(l)
                
            result_data = [1] * len(result_cols)

            result_matrix = sparse.csr_matrix((result_data, (result_rows, result_cols)))

            print("done 4")

            val_answer['tag_to_num'] = val_answer['tags'].map(lambda x: [train_tag_dict[i] for i in x])
            val_answer['songtag'] = val_answer.songs + val_answer.tag_to_num

            answer_rows = []
            for id, st in zip(val_answer.id, val_answer.songtag):
                answer_rows.extend([id] * len(st))

            answer_cols = []
            for l in val_answer.songtag:
                answer_cols.extend(l)
                
            answer_data = [1] * len(answer_cols)

            answer_matrix = sparse.csr_matrix((answer_data, (answer_rows, answer_cols)))

            print("done 6")

            loss = 0
            for i in val.id:
                loss += (len(answer_matrix[i].nonzero()[1]) - result_matrix[i, answer_matrix[i].nonzero()[1]].sum())**2

            with tf.GradientTape() as tape:
                cost_value = loss
            grad_w1,grad_w2,grad_w3,grad_w4,grad_w5,grad_w6,grad_w7,grad_w8,grad_w9,grad_w10,grad_w11,grad_w12,grad_w13,grad_w14,grad_w15,grad_w16,grad_w17,grad_w18,grad_w19 = tape.gradient(cost_value, [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19])
            optimizer = tf.keras.optimizers.SGD(0.01)
            train = optimizer.apply_gradients(grads_and_vars=zip((grad_w1,grad_w2,grad_w3,grad_w4,grad_w5,grad_w6,grad_w7,grad_w8,grad_w9,grad_w10,grad_w11,grad_w12,grad_w13,grad_w14,grad_w15,grad_w16,grad_w17,grad_w18,grad_w19), [w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19]))
            train

            print("Epoch: {:02d}, Cost: {}".format(epoch+1, loss))

        print("Learning Finished!")
        print(w1,w2,w3)
        
    def run(self, train_fname, question_fname, answer_fname):
        print("Loading train file...")
        train_json = load_json(train_fname)
        train_data = pd.read_json(train_fname)

        print("Loading question file...")
        questions = pd.read_json(question_fname)

        print("Loading answer file...")
        answer = pd.read_json(answer_fname)

        print("Generating model...")
        self._train(train_json, train_data, questions, answer)

if __name__ == "__main__":
    fire.Fire(Train)