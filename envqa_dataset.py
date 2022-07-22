# coding=utf-8
from __future__ import print_function
import os
import sys
import re
import json
import time
import pickle
import tqdm
import numpy as np
# import utils
import h5py
import torch
import gzip
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from model.symbol_dict import SymbolDict
from answer_space import answer_space


class VQAFeatureDataset(Dataset):
    def __init__(self, name, annotation=None, img_root='/data/gdf/dvqa/', anno_root='./data/',
                 img_file='env_qa_%s_objects.h5', label_file="env_qa_frame_obj_cls.h5",
                 video_annotation="env_qa_video_annotations.json",
                 predict_segment="env_qa_predicted_segment.json",
                 max_frame=400, max_obj=30, max_q_len=24, max_a_len=7, max_i_len=50, max_event_num=30,
                 batch_size=100, if_gt_vision=False, reset_dict=True):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        self.split = name
        self.max_obj = max_obj
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len   # answer word length
        self.max_i_len = max_i_len   # instruction word length
        self.max_frame = max_frame
        self.max_event_num = max_event_num
        self.batch_size = batch_size
        self.if_gt_vision = if_gt_vision
        self.answer_space = answer_space
        self.anno_root = anno_root
        self.answer_part_order = ['action', 'subjects', 'prep', 'objects', 'state', 'number', 'yes/no', 'order']
        if name != "train":
            reset_dict = False

        # load data
        if not annotation:
            print('loading questions from data/%s_questions.json' % name)
            self._annotation = json.load(open(os.path.join(anno_root, '%s_questions.json' % name)))
        else:
            self._annotation = annotation

        self.instructions = json.load(open(os.path.join(anno_root, "all_instructions.json")))

        self.process_questions()
        self.process_answers()
        self.process_instructions()

        # word_dictionary
        self.dict = {}
        self.get_dictionary(reset_dict)

        img_file = img_file if "%s" not in img_file else img_file % name
        print('loading image features from ' + img_file + ' h5 file')
        hf = h5py.File(os.path.join(img_root, img_file), 'r')
        self.features = hf['features']
        self.bboxes = hf["bboxes"]
        label_path = os.path.join(img_root, label_file)
        print(label_path)
        if os.path.exists(label_path):
            labels = h5py.File(label_path, "r")
            self.labels = labels["labels"]
        else:
            self.raw_labels = hf["labels"][:]
            print("encode object name to word")
            self.objId2wordId, self.labels = self.encode_object_name()
            f = h5py.File(label_path, "w")
            f.create_dataset("labels", self.labels.shape, "f")
            f.close()

        self.v_dim = self.features.shape[2]
        self.bbox_dim = self.bboxes.shape[2]
        self.labels_dim = 1

        self.img_id2pos = json.load(open(os.path.join(anno_root, "img_id2pos.json")))
        self.video_annotations = json.load(open(os.path.join(anno_root, video_annotation)))
        self.predicted_segment = json.load(open(os.path.join(anno_root, predict_segment)))
        # self.video2img_pos = json.load(open(os.path.join(anno_root, "img_id2pos.json"), "r"))
        # self.video2img_pos = {k.lower(): v for k, v in self.video2img_pos.items()}
        print('loading completed')

        # one-hot object label and attribute label for every bbox
        if if_gt_vision:
            # TODO: gt vision graph
            raise NotImplementedError
        # self.tensorize()

        self.filter_questions()

        self.entries = self._load_dataset()
        self.entry_to_ix = []

    def get_dictionary(self, reset_dict):
        anno_root = self.anno_root
        dict_path = os.path.join(anno_root, "dictionaries.pkl")

        if os.path.exists(dict_path):
            print('loading existing dictionaries from %s' % dict_path)
            self.dict = pickle.load(open(dict_path, 'rb'))
        else:
            if self.split == 'train':
                ques_word_list, ans_word_list, ins_word_list, ans_list = self.get_words()
                self.dict["ques_word_dict"] = SymbolDict(ques_word_list, symbol_type='sentence')
                self.dict["ans_word_dict"] = SymbolDict(ans_word_list, symbol_type='sentence')
                self.dict["ins_word_dict"] = SymbolDict(ins_word_list, symbol_type='sentence')
                self.dict["ans_dict"] = SymbolDict(ans_list, symbol_type='class')
                obj_name_list = json.load(open(os.path.join(anno_root, "dict_object_name.json"), "r"))
                obj_name_list = [self.parse_camel_case(obj) for obj in obj_name_list]
                self.dict["obj_name_dict"] = SymbolDict(obj_name_list, symbol_type="class", add_unknown=False)
                self.dict["obj_word_dict"] = SymbolDict(self.get_obj_words(), symbol_type="class")
                self.dict["obj_name_dict"].addToVocab("<PAD>")

                self.dict["ans_part_dict"] = {}
                for ans_type, candidate in self.answer_space.items():
                    self.dict["ans_part_dict"][ans_type] = SymbolDict(candidate, symbol_type='class', add_unknown=False)

                pickle.dump(self.dict, open(dict_path, 'wb'))
                print('\nwrite dictionary to %s' % dict_path)
            else:
                raise Exception("Please first use training data to generate word dictionary!")

    def filter_questions(self):
        print("%d questions before filtering" % len(self._annotation))
        self._annotation = [d for d in self._annotation if d['video_id'] in self.video_annotations and
                            d["video_id"] in self.predicted_segment and
                            self.get_frame_pos(d["video_id"])]
        print("%d questions after filtering" % len(self._annotation))

    def process_questions(self):
        replace_dict = {",": " , ",
                        u"，": " , ",
                        "?": "",
                        "c d": "cd",
                        "scrubbrush": "scrub brush",
                        "spraybottle": "spray bottle",
                        "on/in": "on or in",
                        "creditcard": "credit card",
                        "peppershaker": "pepper shaker",
                        "handtowelholder": "hand towel holder",
                        "toiletpaper": "toilet paper",
                        "stoveburner": "stove burner",
                        "soapbar": "soap bar",
                        "toiletpaperhanger": "toilet paper hanger",
                        "saltshaker": "salt shaker",
                        "tissuebox": "tissue box",
                        "sidetable": "side table",
                        "butterknife": "butter knife",
                        "papertowelroll": "paper towel roll",
                        "vedio": "video",
                        "remotecontrol": "remote control",
                        "showerdoor": "shower door",
                        "showercurtain": "shower curtain",
                        "wateringcan": "watering can",
                        "garbagecan": "garbage can",
                        "teddybear": "teddy bear",
                        "tennisracket": "tennis racket",
                        "alarmclock": "alarm clock",
                        "showerglass": "shower glass",
                        "tvstand": "tv stand",
                        "towelholder": "towel holder",
                        "vidio": "video",
                        "dishsponge": "dish sponge",
                        "soapbottle": "soap bottle",
                        "garbagepen": "garbage can",
                        "diningtable": "dining table",
                        "winebottle": "wine bottle",
                        "not-used": "not used",
                        "handtowel": "hand towel",
                        "yelloe": "yellow",
                        "baseballbat": "baseball bat",
                        "yelloww": "yellow",
                        "garbages": "garbage",
                        "paperhanger": "paper hanger",
                        "galss": "glass",
                        "toppest": "topest",
                        "in/on": "in or on",
                        "onhte": "on the",
                        "bowl?": "bowl"
                        }
        for anno in self._annotation:
            for k, v in replace_dict.items():
                anno['question'] = anno['question'].replace(k, v)
            anno['question'] = anno['question'].lower()
            anno['question'] = re.sub(r'[\*\)\(\?\.]', '', anno['question'])

    def process_answers(self):
        replace_dict = {",": " , ",
                        u"，": " , ",
                        "(": " (",
                        "c d": "cd",
                        "t v": "tv",
                        "don't": "do not",
                        "doesn't": "does not",
                        "hasn't": "has not",
                        "it's": "it is",
                        "scrubbrush": "scrub brush",
                        "spraybottle": "spray bottle",
                        "on/in": "on or in",
                        "creditcard": "credit card",
                        "peppershaker": "pepper shaker",
                        "handtowelholder": "handtowel holder",
                        "toiletpaper": "toilet paper",
                        "stoveburner": "stove burner",
                        "soapbar": "soap bar",
                        "toiletpaperhanger": "toilet paper hanger",
                        "saltshaker": "salt shaker",
                        "tissuebox": "tissue box",
                        "sidetable": "side table",
                        "butterknife": "butter knife",
                        "papertowelroll": "paper towel roll",
                        "vedio": "video",
                        "remotecontrol": "remote control",
                        "showerdoor": "shower door",
                        "showercurtain": "shower curtain",
                        "wateringcan": "watering can",
                        "garbagecan": "garbage can",
                        "teddybear": "teddy bear",
                        "tennisracket": "tennis racket",
                        "alarmclock": "alarm clock",
                        "showerglass": "shower glass",
                        "tvstand": "tv stand",
                        "towelholder": "towel holder",
                        "vidio": "video",
                        "dishsponge": "dish sponge",
                        "soapbottle": "soap bottle",
                        "garbagepen": "garbage can",
                        "diningtable": "dining table",
                        "winebottle": "wine bottle",
                        "not-used": "not used",
                        "handtowel": "hand towel",
                        "handtowelroll": "hand towel roll",
                        }
        for anno in self._annotation:
            for k, v in replace_dict.items():
                anno['answer'] = anno['answer'].replace(k, v)
            anno['answer'] = anno['answer'].lower()
            anno['answer'] = re.sub(r'[\*\)\(\?\.]', '', anno['answer'])

    def answer_to_role_value(self, answer):
        answer = " " + answer + " "
        new_answer = {}
        for ans_type, candidates in self.answer_space.items():
            new_answer[ans_type] = "NONE"
            target_index = []
            for candidate in candidates:
                flag, index = self.if_in_answer(candidate, answer)
                if flag:
                    if ans_type == "objects":
                        target_index.append((index + len(candidate), len(candidate), candidate))
                    else:
                        target_index.append((index, -len(candidate), candidate))

            if target_index:
                if ans_type == "objects":
                    if len(target_index) > 1:
                        new_answer[ans_type] = sorted(target_index)[-1][2]
                elif ans_type == "prep":
                    if target_index:
                        if target_index[0][2] == "on" and "turning on" in answer:
                            new_answer[ans_type] = "NONE"
                        elif target_index[0][2] == "off" and "turning off" in answer:
                            new_answer[ans_type] = "NONE"
                        else:
                            new_answer[ans_type] = sorted(target_index)[0][2]
                else:
                    new_answer[ans_type] = sorted(target_index)[0][2]

        if new_answer["subjects"] == new_answer["objects"]:
            new_answer["objects"] = "NONE"

        return new_answer

    @staticmethod
    def if_in_answer(candidate, answer):
        candidate = " " + candidate + " "
        flag = True
        index = 10000
        if len(candidate.split()) > 1 and " " + "".join(candidate.split()) + " " in answer:
            index = answer.index("".join(candidate.split()))
        else:
            for w in candidate.split():
                if w not in answer.split():
                    flag = False
                else:
                    if index > answer.index(w):
                        index = answer.index(w)

        return flag, index

    @staticmethod
    def encode_former_later(answer, question):
        candidate = []
        for part in question.split(","):
            if " or " in part:
                candidate = part.split(" or ")
                candidate = [c.strip() for c in candidate]

        if not candidate:
            return None

        if answer == candidate[0]:
            out = "former"
        elif answer == candidate[1]:
            out = "latter"
        elif "unanswerable" in answer or "nothing" in answer:
            out = "unanswerable"
        else:
            ans_set = set(answer.split())
            candidate_set = [len(set(c.split()) & ans_set) for c in candidate]
            out = "former" if candidate_set[0] > candidate_set[1] else "latter"

        return out

    @staticmethod
    def decode_former_later(answer, question):
        candidate = []
        for part in question.split(","):
            if " or " in part:
                candidate = part.split(" or ")
                candidate = [c.strip() for c in candidate]

        if len(candidate) < 1:
            out = "unanswerable"
        elif len(candidate) > 1:
            out = candidate[0] if answer == "former" else candidate[1]
        else:
            out = candidate[0]

        return out

    def process_instructions(self):
        for env_id, sentences in self.instructions.items():
            for i, sentence in enumerate(sentences):
                sentences[i] = sentence.replace(",", " , ")

    def get_words(self):
        ques_words = {}
        ans_words = {}
        ins_words = {"<SEP>": True}
        answers = {}
        for anno in self._annotation:
            for w in anno['question'].split(" "):
                ques_words[w.lower()] = True
            for w in anno['answer'].split(" "):
                ans_words[w.lower()] = True
            answers[anno['answer']] = True

        for env_id, sentences in self.instructions.items():
            for sentence in sentences:
                for w in sentence.split():
                    ins_words[w] = True

        return [w for w in ques_words], [w for w in ans_words], [w for w in ins_words], [ans for ans in answers]

    def get_obj_words(self):
        obj_words = {}
        for obj_name in self.dict["obj_name_dict"].id2sym:
            for w in obj_name.split():
                obj_words[w] = True
        return [w for w in obj_words]

    def encode_object_name(self):
        # turn object name index to object word sequence, e.g., 1 (CounterTop) -> [2, 3, 0] counter top
        objId2wordId = {}
        for idx, name in enumerate(self.dict["obj_name_dict"].id2sym):
            name_token = self.dict["obj_word_dict"].encodeSeq(name.split())
            objId2wordId[idx] = np.array(name_token + [0] * (3 - len(name_token)))   # max number of object word is 3

        new_labels = np.zeros([self.raw_labels.shape[0], self.raw_labels.shape[1], 3])   # (img_num, obj_num, 3)
        for img_ix, raw in enumerate(tqdm.tqdm(self.raw_labels)):
            obj_num = np.sum(np.sum(self.bboxes[img_ix], -1) > 0)
            for j, cls in enumerate(raw):
                if j < obj_num:
                    # if cls == -1, it is global image feature
                    new_labels[img_ix][j] = objId2wordId[cls] if cls >= 0 else np.array([0, 0, 0])

        return objId2wordId, new_labels

    def tokenize_and_encode(self, sentence, sentence_type="question"):
        if sentence_type == "question":
            words = sentence.lower().split(" ")
            words = self.dict["ques_word_dict"].encodeSeq(words)
            # right align
            tokens = [0] * (self.max_q_len - len(words)) + words if len(words) <= self.max_q_len else words[:self.max_q_len]
        elif sentence_type == "answer":
            words = sentence.lower().split(" ")
            words = self.dict["ans_word_dict"].encodeSeq(words, addStart=True, addEnd=True)
            # left align
            tokens = words + [0] * (self.max_a_len - len(words)) if len(words) <= self.max_a_len else words[:self.max_a_len]
        elif sentence_type == "instruction":
            assert type(sentence) is list
            words = " <SEP> ".join(sentence).split()
            words = self.dict["ins_word_dict"].encodeSeq(words, addStart=True, addEnd=True)
            # left align
            # tokens = words + [0] * (self.max_i_len - len(words)) if len(words) <= self.max_i_len else words[:self.max_i_len]
            tokens = self.padding_sequence(words, self.max_i_len, left_align=True)
        else:
            raise NotImplementedError

        return tokens


    @staticmethod
    def padding_sequence(seq, length, left_align=True):
        if left_align:
            out = seq + [0] * (length - len(seq)) if len(seq) <= length else seq[:length]
        else:
            out = [0] * (length - len(seq)) if len(seq) <= length else seq[:length] + seq
        return out

    # def get_mask(self, object_ids, object_idtoix):
    #     mask = [0] * self.max_obj
    #     for obj in object_ids:
    #         mask[object_idtoix[obj]] = 1
    #     return mask

    def get_instruction(self, question_id):
        video_id = self.get_video_id(question_id, if_lower=False)
        instruction = self.instructions[video_id]
        return self.tokenize_and_encode(instruction, "instruction")

    def get_part_answer(self, answer, question):
        if ("later" in question or "first" in question) and " or " in question:
            answer = self.encode_former_later(answer, question)

        ans_part = self.answer_to_role_value(answer)
        ans_part_encoded = [self.dict["ans_part_dict"][part].encodeSym(ans_part[part])
                            for part in self.answer_part_order]
        return ans_part_encoded

    def _load_dataset(self):
        entries = []
        print("load dataset")
        for ix, question in enumerate(tqdm.tqdm(self._annotation)):
            # video_id = str(question['video_id'])
            # if len(self.obj_list[img_id]) > 36:  # or question['functions'][-1]['function'] != 'Count':
            #     continue
            # else:
            entries.append(self._create_entry(question))
        return entries

    def _create_entry(self, question):
        entry = {'video_id': question['video_id'],
                 'question': self.tokenize_and_encode(question['question'], "question"),
                 'answer_words': self.tokenize_and_encode(question['answer'], "answer"),
                 'answer_part': self.get_part_answer(question['answer'], question["question"]),
                 'answer': self.dict["ans_dict"].encodeSym(question['answer']),
                 'question_id': question['question_id'],
                 "instruction": self.get_instruction(question['question_id'])}
        return entry

    def get_frame_pos(self, video_id):
        # index of frame feature in h5 file
        frame_pos = []
        for f_id in self.video_annotations[video_id]["frame_ids"]:
            if self.img_id2pos.get(f_id, None) is not None:
                frame_pos.append(self.img_id2pos[f_id])
        # if len(frame_pos) == 0:
        #     print("%s has no images" % video_id)
        frame_pos = frame_pos[:min(len(frame_pos), self.max_frame)]
        return frame_pos

    @staticmethod
    def get_video_id(question_id, if_lower=True):
        if if_lower:
            video_id = '_'.join(question_id.split('_')[:3]).lower()
        else:
            video_id = '_'.join(question_id.split('_')[:3])
        return video_id

    def get_video_feature(self, video_id, frame_num=None):
        frame_num = frame_num if frame_num else self.max_frame
        out_feat = np.zeros([frame_num, self.max_obj, self.v_dim])
        frame_pos = self.get_frame_pos(video_id)
        if frame_pos:
            img_feat = self.features[frame_pos]
            frame_count, _, _ = img_feat.shape
            if frame_count < frame_num:
                out_feat[:frame_count] = img_feat
            else:
                out_feat = img_feat[:frame_num]
        return out_feat

    def get_bbox_feature(self, video_id, frame_num=None):
        frame_num = frame_num if frame_num else self.max_frame
        out_feat = np.zeros([frame_num, self.max_obj, self.bbox_dim])
        frame_pos = self.get_frame_pos(video_id)
        bbox_feat = self.bboxes[frame_pos]
        frame, _, _ = bbox_feat.shape
        if frame < frame_num:
            out_feat[:frame] = bbox_feat
        else:
            out_feat = bbox_feat[:frame_num]
        return out_feat

    def get_cls_feature(self, video_id, frame_num=None):
        frame_num = frame_num if frame_num else self.max_frame
        out_feat = np.zeros([frame_num, self.max_obj, 3])
        frame_pos = self.get_frame_pos(video_id)
        obj_idx = self.labels[frame_pos]

        frame, _, _ = obj_idx.shape
        if frame < frame_num:
            out_feat[:frame] = obj_idx
        else:
            out_feat = obj_idx[:frame_num]
        return out_feat

    def get_event_segment_feature(self, video_id, frame_num=None, event_num=None):
        frame_num = frame_num if frame_num else self.max_frame
        event_num = event_num if event_num else self.max_event_num

        event_feat = np.zeros([frame_num, event_num])
        for i, seg in enumerate(self.predicted_segment[video_id]["annotations"]):
            if i < event_num:
                start_ix, end_ix = seg["segment_frame"]
                if end_ix != start_ix:
                    length = end_ix - start_ix
                    # try:
                    event_feat[start_ix:end_ix, i] = 1.0 / length
                # except:
                #     import IPython.core.debugger
                #     dbg = IPython.core.debugger
                #     dbg.set_trace()
        return event_feat

    def get_focus_feature(self, video_id, bin_size=30, img_width=400, frame_num=None, object_num=None):
        frame_num = frame_num if frame_num else self.max_frame
        object_num = object_num if object_num else self.max_obj
        frame_pos = self.get_frame_pos(video_id)
        video_bboxes = self.bboxes[frame_pos]

        attention = np.zeros([frame_num, object_num])
        for frame_ix, video_bbox in enumerate(video_bboxes):
            for obj_ix, bbox in enumerate(video_bbox):
                center = self.get_object_center(bbox)
                # if the object center in the centric bin, the objects are attended
                if img_width / 2 - bin_size < center < img_width / 2 + bin_size:
                    attention[frame_ix, obj_ix] = 1
                else:
                    attention[frame_ix, obj_ix] = 0
        return attention

    @staticmethod
    def get_object_center(bbox):
        return (bbox[0] + bbox[2]) / 2

    @staticmethod
    def parse_camel_case(sentence):
        sentence = sentence.split()
        out_sentence = []
        for text in sentence:
            out = ""
            for index, char in enumerate(text):
                if char.isupper() and index != 0:
                    out += " "
                out += char.lower()
            out_sentence.append(out)
        out = " ".join(out_sentence)
        out = out.replace("c d", "cd")
        out = out.replace("t v", "tv")
        out = out.replace("< p a d>", "<PAD>")
        return out

    def decode_data(self, data, data_type, question=None):
        if data_type == "question":
            return self.dict["ques_word_dict"].decodeSeq(data)
        elif data_type == "answer_words":
            return self.dict["ans_word_dict"].decodeSeq(data)
        elif data_type == "answer":
            return self.dict["ans_dict"].decodeId(torch.argmax(data))
        elif data_type == "answer_part":
            results = []
            for i, part in enumerate(self.answer_part_order):
                results.append(self.dict["ans_part_dict"][part].decodeId(data[i]))
            return results
            # out = " ".join([p for p in results if p != "NONE"])
            # if out in ["former", "latter"]:
            #     out = self.decode_former_later(out, question)
            # return out
        else:
            return data

    def __getitem__(self, index):
        # image, question, answer, functions, ground truth of every function
        # entry = self.entries[self.entry_to_ix[index]]
        batch = {}
        entry = self.entries[index]
        batch['question'] = torch.from_numpy(np.array(entry['question'])).type(torch.LongTensor)  # [max_question_num]
        batch['question_id'] = entry['question_id']
        batch['question_mask'] = (batch['question'] != 0).type(torch.FloatTensor)  # [max_question_num]

        # append object label and attribute label to visual features
        batch['video_id'] = entry['video_id']
        video_id = batch["video_id"]
        if self.if_gt_vision:
            # TODO:
            raise NotImplementedError
        else:
            features = self.get_video_feature(video_id)
        batch['video'] = torch.from_numpy(features).type(torch.FloatTensor)  # [frame, obj_num, feat_dim]
        batch['video_event_att'] = torch.from_numpy(self.get_event_segment_feature(video_id)).type(torch.FloatTensor)
        batch['video_focus_att'] = torch.from_numpy(self.get_focus_feature(video_id)).type(torch.FloatTensor)
        batch['bbox'] = torch.from_numpy(self.get_bbox_feature(video_id)).type(torch.FloatTensor)
        batch['object_cls'] = torch.from_numpy(self.get_cls_feature(video_id)).type(torch.LongTensor)
        batch['video_mask'] = (torch.sum(batch['video'], dim=-1) != 0).type(torch.FloatTensor)  # [frame, obj_num]

        batch['answer'] = torch.zeros(self.dict["ans_dict"].getNumSymbols())
        batch['answer'][entry['answer']] = 1  # [answer_vocab]
        batch['answer_part'] = torch.tensor(entry["answer_part"]).type(torch.LongTensor)
        batch['answer_words'] = torch.from_numpy(np.array(entry['answer_words'])).type(torch.LongTensor)

        batch['instruction'] = torch.from_numpy(np.array(entry['instruction'])).type(torch.LongTensor)
        return batch

    def __len__(self):
        return len(self.entries)


class BatchSampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data
