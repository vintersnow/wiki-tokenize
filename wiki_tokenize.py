from nltk.tokenize import word_tokenize, sent_tokenize
import json
from collections import defaultdict
import queue
from threading import Thread
from collections import Counter
from functools import reduce
import time
import glob
from argparse import ArgumentParser
from os import path

parser = ArgumentParser('python train.py')

# Data
parser.add_argument('--data_path', type=str,
                    default='./wikidump/extracted/*/wiki_*',
                    help='path to data directories')
parser.add_argument('--out_path', type=str,
                    default='./wikidump/',
                    help='path to output directory')
parser.add_argument('--time_wait', type=int,
                    default=60,
                    help='sleep time for check isfinish')

args = parser.parse_args()


def tokenize(obj):
    if isinstance(obj, str):
        return word_tokenize(obj)
    else:
        raise ValueError('Not supported instance. %s' % type(obj))


class Tokenizer(object):
    JSON_Q_SIZE = 1000
    TOKN_Q_SIZE = 1000

    def __init__(self):

        self._json_q = queue.Queue(self.JSON_Q_SIZE)
        self._tokn_q = queue.Queue(self.TOKN_Q_SIZE)

        self._finished = False

        self._tokn_th_n = 8
        self._gather_th_n = 4
        self._vocabs = [
            defaultdict(lambda: 0) for _ in range(self._gather_th_n)
        ]

        self.make_thread([], self.readjson)
        self._token_threads = []
        for _ in range(self._tokn_th_n):
            self.make_thread(self._token_threads, self.tokenize)
        self._gatehr_threads = []
        for i in range(self._gather_th_n):
            self.make_thread(self._gatehr_threads, self.gather,
                             (self._vocabs[i], i))

    def make_thread(self, threads_list, target, args=()):
        threads_list.append(Thread(target=target, args=args))
        threads_list[-1].daemon = True
        threads_list[-1].start()

    def readjson(self):
        file_list = glob.glob(args.data_path)
        for filename in file_list:
            with open(filename, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self._json_q.put(data['text'])

        self._finished = True

    def tokenize(self):
        while True:
            text = self._json_q.get()
            for sent in sent_tokenize(text):
                tok = word_tokenize(sent)
                self._tokn_q.put(tok)

    def gather(self, vocab, id):
        print('start gather %d' % id)
        output = open(path.join(args.out_path, 'output-%d.txt' % id), 'w')
        while True:
            # if id == 0:
            #     print('q:', self._json_q.qsize(), self._tokn_q.qsize())
            if self.isfinish():
                break

            tok = self._tokn_q.get()
            for t in tok:
                vocab[t] += 1
            output.write(' '.join(tok) + '\n')

        output.close()
        print('end gather %d' % id)

    def output_vocab(self):
        vocab_f = open(path.join(args.out_path, 'vocab'), 'w')
        vocab = reduce(lambda x, y: x + y, (Counter(d) for d in self._vocabs))
        print(vocab)
        for k, v in sorted(vocab.items(), key=lambda x: -x[1]):
            vocab_f.write('%s %d\n' % (k, v))
        vocab_f.close()

    # def close(self):
    # self._output.close()

    def isfinish(self):
        return self._finished and self._json_q.qsize(
        ) == 0 and self._tokn_q.qsize() == 0


def origin():
    vocab = defaultdict(lambda: 0)
    vocab_f = open(path.join(args.out_path, 'vocab'), 'w')
    output = open(path.join(args.out_path, 'output.txt'), 'w')
    file_list = glob.glob(args.data_path)
    for filename in file_list:
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                for sent in sent_tokenize(data['text']):
                    # print('sent', sent)
                    tok = tokenize(sent)
                    for t in tok:
                        vocab[t] += 1
                    output.write(' '.join(tok) + '\n')

    for k, v in sorted(vocab.items(), key=lambda x: -x[1]):
        vocab_f.write('%s %d\n' % (k, v))

    output.close()
    vocab_f.close()


if __name__ == '__main__':
    # origin()
    tok = Tokenizer()
    elapsed = 0
    while not tok.isfinish():
        time.sleep(60)
        elapsed += 1
        print('elapsed:', elapsed)

    tok.output_vocab()
