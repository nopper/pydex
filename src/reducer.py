#!/usr/bin/env python

from mpi4py import MPI
from bisect import insort, bisect
from heapq import heappush, heappop
from tempfile import NamedTemporaryFile
from collections import defaultdict

from struct import calcsize, pack
from tags import *
from logger import get_logger

log = get_logger('reducer')

rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

status = MPI.Status()
comm = MPI.COMM_WORLD

class Reducer(object):
    def __init__(self, output_path, num_mappers):
        log.info("[0] Reducer at %s (rank=%d)" % (name, rank))

        self.output_path = output_path
        self.num_workers = num_mappers

        log.info("[1] Starting")
        self.reduce_word_count()
        log.info("[1] Finished")
        comm.Barrier()

        log.info("[2] Starting")
        self.reduce_word_count_per_doc()
        log.info("[2] Finished")
        comm.Barrier()

        log.info("[3] Starting")
        self.reduce_words_per_doc()
        log.info("[3] Finished")
        comm.Barrier()

    def reduce_words_per_doc(self):
        remaining = self.num_workers

        heap = []
        words_dct = defaultdict(int)
        words_length = 0
        threshold = 1024 * 1024

        while remaining > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)

            if msg == MSG_COMMAND_QUIT:
                remaining -= 1
                log.debug("Received termination message from %d" % \
                        status.Get_source())
            else:
                word, doc_id, word_count, word_per_doc = msg

                heappush(heap, msg)

                words_dct[word] += 1
                words_length += len(word)

                if words_length > threshold or remaining == 0:
                    self.write_words_per_doc(heap, words_dct)
                    words_dct = {}
                    words_length = 0

        self.write_words_per_doc(heap, words_dct)

    def write_words_per_doc(self, heap, words_dct):
        handle = NamedTemporaryFile(prefix='reduce-3-chunk-',
                                    dir=self.output_path, delete=False)

        while heap:
            word, doc_id, word_count, word_per_doc = heappop(heap)
            handle.write("%s %d %d %d %d\n" % (word, doc_id, word_count,
                                               word_per_doc, words_dct[word]))

        handle.close()

    def reduce_word_count_per_doc(self):
        remaining = self.num_workers

        # We need to create word, doc-id <-> wordCount, wordPerDoc
        # At this point we are going to create an heap to keep order of the words.
        # A dictionary on the other hand will keep track of
        # dict[doc_id] = (word_count, {word1:cnt, word2:cnt ..})

        num_docs = 0
        num_words = 0
        words_length = 0

        doc_dict = {}
        docid_list = []
        words_list = []

        threshold = 1024 * 1024 # 1 MByte

        while remaining > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)

            if msg == MSG_COMMAND_QUIT:
                remaining -= 1
                log.debug("Received termination message from %d" % \
                        status.Get_source())
            else:
                doc_id, word, word_count = msg

                if not doc_id in doc_dict:
                    counter = [0, {}]
                    doc_dict[doc_id] = counter

                    num_docs += 1
                    insort(docid_list, doc_id)
                else:
                    counter = doc_dict[doc_id]

                counter[0] += word_count

                if not word in counter[1]:
                    counter[1][word] = word_count

                    num_words += 1
                    words_length += len(word)
                    pos = bisect(words_list, word)

                    if not words_list or words_list[max(0, pos - 1)] != word:
                        words_list.insert(pos, word)
                else:
                    counter[1][word] += word_count

            # This is somehow dummy
            if words_length > threshold or remaining == 0:
                self.write_word_count_per_doc(docid_list, words_list, doc_dict)

                doc_dict = {}
                docid_list = []
                words_list = []

                num_docs = 0
                num_words = 0
                words_length = 0

    def write_word_count_per_doc(self, docid_list, words_list, doc_dict):
        handle = NamedTemporaryFile(prefix='reduce-2-chunk-',
                                    dir=self.output_path, delete=False)

        for docid in docid_list:
            counter = doc_dict[docid]

            for word in words_list:
                if word not in counter[1]:
                    continue

                handle.write("%d %s %d %d\n" % \
                    (docid, word, counter[1][word], counter[0])
                )

        handle.close()

    def reduce_word_count(self):
        heap = []
        length = 0
        threshold = 1024 * 1024 # 1 MByte
        remaining = self.num_workers

        while remaining > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)

            if msg == MSG_COMMAND_QUIT:
                remaining -= 1
                log.debug("Received termination message from %d" % \
                        status.Get_source())
            else:
                word, doc_id = msg

                length += len(str(doc_id)) + len(word) + 1
                heappush(heap, msg)

            if length > threshold or remaining == 0:
                self.write_partition(heap)
                heap = []
                length = 0

    def write_partition(self, heap):
        if not heap:
            return

        handle = NamedTemporaryFile(prefix='reduce-chunk-',
                                    dir=self.output_path, delete=False)
        counter = 1
        last_word, last_doc_id = heappop(heap)
        word, doc_id = None, None

        while heap:
            word, doc_id = heappop(heap)

            if last_doc_id == doc_id and last_word == word:
                counter += 1
            else:
                handle.write("%s %d %d\n" % (last_word, last_doc_id, counter))
                counter = 1

            last_word, last_doc_id = word, doc_id

        handle.write("%s %d %d\n" % (word, doc_id, counter))
        handle.close()

        log.debug("Wrote partition as %s" % handle.name)
