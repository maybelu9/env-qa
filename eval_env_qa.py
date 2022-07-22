from answer_space import answer_space
import tqdm
import re
import json


class EnvQAEval:
    def __init__(self, envQA, envQARes):
        '''
        envQA: [{"question_id": str, "answer": str, "question": str}] Ground Truth File
        envQARes: [{"question_id": str, "answer": str, dict or list}, ...] Prediction Results
        answer could be in following format:
        sentence format answer: [{"question_id": str, "answer": str}, ....]
        role value dict format answer: [{"question_id": str, "answer": {"object": str, "subject": str, ...}}, ...]
        list of value format answer: [{"question_id": str, "answer": list}], list the vale of roles in following order
                                      ['action', 'subjects', 'prep', 'objects', 'state', 'number', 'yes/no', 'order']
        '''
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        if not envQA:
            envQA = json.load(open("data/test_full_question.json"))
        self.envQA = {d["question_id"]: d for d in envQA}
        self.envQARes = {d["question_id"]: d for d in envQARes}
        self.params = {'image_id': [d['question_id'] for d in envQA]}
        self.answer_space = answer_space
        self.scores = {}

    def evaluate(self):
        scores = {}
        for q_id in tqdm.tqdm(self.envQA):
            gt = self.envQA[q_id]
            res = self.envQARes.get(q_id, {"question_id": q_id, "answer": ""})

            gt_rv = self.get_role_value(gt["answer"], gt["question"])
            res_rv = self.get_role_value(res["answer"], gt["question"])

            inter, union = 0, 0
            for role in gt_rv:
                if gt_rv[role] == res_rv[role] != "NONE":
                    inter += 1

                if gt_rv[role] != "NONE" or res_rv[role] != "NONE":
                    union += 1
            scores[q_id] = (float(inter) + 1e-6) / (float(union) + 1e-6)
        print("%.4f" % (sum([s for _, s in scores.items()]) / len(scores)))
        self.scores = scores

    def get_role_value(self, answer, question):
        # we support evaluation of three types of answer format
        if type(answer) is str:
            if ("later" in question or "first" in question) and " or " in question:
                answer = self.encode_former_later(answer, question)

            new_answer = self.answer_to_role_value(answer)
        elif type(answer) is list:
            # please ensure the roles in the answer is in the following order
            role_order = ['action', 'subjects', 'prep', 'objects', 'state', 'number', 'yes/no', 'order']
            new_answer = {}
            for i, role in enumerate(role_order):
                new_answer[role] = answer[i]
        else:
            new_answer = answer
        return new_answer

    def answer_to_role_value(self, answer):
        # turn a sentence answer to role value format
        answer = self.process_answers(answer)
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
        # for querying order question, we simply let model to choosee the former or latter option by answering "former" or "latter".
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

    def process_answers(self, answer):
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
        for k, v in replace_dict.items():
            answer = answer.replace(k, v)
        answer = answer.lower()
        answer = re.sub(r'[\*\)\(\?\.]', '', answer)
        return answer

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

    # def EvalAccuracy(self):

