#!/usr/bin/env python

from mpi4py import MPI
from heapq import heappush, heappop
from tempfile import NamedTemporaryFile
from collections import OrderedDict

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
        log.info("Started a new reducer on %s (rank=%d)" % (name, rank))

        self.output_path = output_path
        self.num_workers = num_mappers

        self.reduce_word_count()
        comm.Barrier()

        self.reduce_word_count_per_doc()
        comm.Barrier()

    def reduce_word_count_per_doc(self):
        remaining = self.num_workers

        # We need to create word, doc-id <-> wordCount, wordPerDoc
        # At this point we are going to create an heap to keep order of the words.
        # A dictionary on the other hand will keep track of
        # dict[doc_id] = (word_count, {word1:cnt, word2:cnt ..})

        num_docs = 0
        num_words = 0
        words_length = 0

        word_heap = []
        doc_dict = {}

        while remaining > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)

            if msg == MSG_COMMAND_QUIT:
                remaining -= 1
            else:
                try:
                    doc_id, word, word_count = msg
                except Exception, exc:
                    print "MSG FROM", status.Get_source(), status.Get_tag()
                    print msg
                    raise exc

                if not word in word_heap:
                    heappush(word_heap, word)
                    num_words += 1
                    words_length += len(word)

                if not doc_id in doc_dict:
                    doc_dict[doc_id] = (0, defaultdict(int))
                    num_docs += 1

                doc_tuple = doc_dict[doc_id]
                doc_tuple[0] += word_count
                doc_tuple[1][word] += word_count

            # This is somehow dummy
            if words_length > threshold or remaining == 0:
                self.write_word_count_per_doc(word_heap, doc_dict)

                doc_dict = {} # Or maybe clear?
                num_docs = 0
                num_words = 0
                words_length = 0

    def write_word_count_per_doc(self, word_heap, doc_dict):
        handle = NamedTemporaryFile(prefix='reduce-2-chunk-',
                                    dir=self.output_path, delete=False)

        doc_ids = sorted(doc_dict.keys())

        while not word_heap:
            word = heappop(word_heap)

            for doc_id in doc_ids:
                dct = doc_dict[doc_id]
                doc_word_count = dct[0]
                word_count     = dct[1][word]

                handle.write("%s %d %d %d\n" % \
                    (word, doc_id, word_count, doc_word_count)
                )

    def reduce_word_count(self):
        heap = []
        length = 0
        threshold = 1024 * 1024 # 1 MByte
        remaining = self.num_workers

        while remaining > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)

            if msg == MSG_COMMAND_QUIT:
                remaining -= 1
                log.info("Received termination message from %d" % \
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

        log.info("Wrote partition as %s" % handle.name)
