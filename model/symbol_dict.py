import numpy as np
import math


class SymbolDict(object):
    def __init__(self, symbol_list, symbol_type='word', global_index=True, dict_name='', add_unknown=True):
        self.symbol_list = symbol_list
        self.symbol_type = symbol_type
        self.dict_name = dict_name

        self.padding = "<PAD>"
        self.unknown = "<UNK>"
        self.start = "<START>"
        self.end = "<END>"

        self.invalidSymbols = [self.padding, self.unknown, self.start, self.end]

        self.sym2id = {}
        self.grp2id = {}
        self.id2grp = []
        self.id2sym = []
        self.allSeqs = []

        self.OOV = {}

        # for group word, index of a symbol is the position in its group or in the vocab
        self.global_index = global_index
        self.creatVocab(add_unknown)

    def getNumSymbols(self):
        return len(self.sym2id)

    def getNumGroups(self):
        return len(self.grp2id)

    def isValid(self, enc):
        return enc not in self.invalidSymbols

    def isStop(self, enc):
        return enc == self.end

    def resetSeqs(self):
        self.allSeqs = []

    def addSymbols(self, seq):
        if type(seq) is not list:
            seq = [seq]
        self.allSeqs += seq

    # Call to create the words-to-integers vocabulary after (reading word sequences with addSymbols).
    def addToVocab(self, symbol):
        if symbol not in self.sym2id:
            self.sym2id[symbol] = self.getNumSymbols()
            self.id2sym.append(symbol)

    def addGroup(self, group_name, concepts, global_index=True):
        group_id = len(self.grp2id)
        self.grp2id[group_name] = group_id
        self.id2grp.append(group_name)

        self.id2sym.append({})
        for i, concept in enumerate(concepts):
            ix = len(self.sym2id) if global_index else i
            self.sym2id[concept] = (group_id, ix)
            self.id2sym[group_id][ix] = concept

    def creatVocab(self, add_unknown=True):
        if self.symbol_type == 'class':
            self.sym2id = {self.unknown: 0} if add_unknown else {}
            self.id2sym = [self.unknown] if add_unknown else []
            for i, symbol in enumerate(self.symbol_list):
                self.addToVocab(symbol)

        elif self.symbol_type == 'word':
            self.sym2id = {self.padding: 0, self.unknown: 1}
            self.id2sym = [self.padding, self.unknown]
            for i, symbol in enumerate(self.symbol_list):
                self.addToVocab(symbol)

        elif self.symbol_type == 'group_word':
            self.sym2id = {self.padding: (0, 0), self.unknown: (1, 1)}
            self.grp2id = {self.padding: 0, self.unknown: 1}
            self.id2grp = [self.padding, self.unknown]
            self.id2sym = [{0: self.padding}, {1: self.unknown}]

            for group, symbols in self.symbol_list.items():
                self.addGroup(group, symbols, self.global_index)

        elif self.symbol_type == 'sentence':
            self.sym2id = {self.padding: 0, self.unknown: 1, self.start: 2, self.end: 3}
            self.id2sym = [self.padding, self.unknown, self.start, self.end]
            for i, symbol in enumerate(self.symbol_list):
                self.addToVocab(symbol)


    # Encodes a symbol. Returns the matching integer.
    def encodeSym(self, symbol):
        if symbol not in self.sym2id:
            if not self.OOV.get(symbol, False):
                # print(self.dict_name, symbol)
                pass
            self.OOV[symbol] = self.OOV.get(symbol, 0) + 1
            symbol = self.unknown
        return self.sym2id[symbol]  # self.sym2id.get(symbol, None) # # -1 VQA MAKE SURE IT DOESNT CAUSE BUGS

    '''
    Encodes a sequence of symbols.
    Optionally add start, or end symbols. 
    Optionally reverse sequence 
    '''

    def encodeSeq(self, decoded, addStart=False, addEnd=False, reverse=False):
        if reverse:
            decoded.reverse()
        if addStart:
            decoded = [self.start] + decoded
        if addEnd:
            decoded = decoded + [self.end]
        encoded = [self.encodeSym(symbol) for symbol in decoded]
        return encoded

    # Decodes an integer into its symbol
    def decodeId(self, enc):
        if self.symbol_type == 'group_word':
            if enc[0] < self.getNumGroups():
                if enc[1] < len(self.id2sym[enc[0]]):
                    return self.id2sym[enc[0]][enc[1]]
            return self.unknown
        else:
            return self.id2sym[enc] if enc < self.getNumSymbols() else self.unknown

    '''
    Decodes a sequence of integers into their symbols.
    If delim is given, joins the symbols using delim,
    Optionally reverse the resulted sequence 
    '''

    def decodeSeq(self, encoded, delim=None, reverse=False, stopAtInvalid=True):
        if self.symbol_type == 'sentence':
            decoded = []
            for enc in encoded:
                if self.isValid(self.decodeId(enc)):
                    decoded.append(self.decodeId(enc))
                elif self.isStop(self.decodeId(enc)):
                    break
                else:
                    continue

            if reverse:
                decoded.reverse()

            if delim is not None:
                return delim.join(decoded)

            return decoded
        else:
            decoded = []
            for enc in encoded:
                decoded.append(self.decodeId(enc))
            return decoded
            # print('not support decode sequence of word symbols')

    def decodeMat(self, encoded):
        decoded = []
        for line in encoded:
            decoded.append(self.decodeSeq(line))
        return decoded

    def __len__(self):
        return len(self.sym2id)
