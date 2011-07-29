#!/usr/bin/env python

import os
import sys
import mmap
import math
import contextlib

from tags import *
from time import sleep
from mpi4py import MPI
from logger import get_logger
from extractor import DocumentExtractor

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

log = get_logger('mapper-%d' % rank)

class Mapper(object):
    def __init__(self, reducers):
        log.info("[0] Mapper at %s (rank=%d)" % (name, rank))

        self.tasks = 0
        self.reducers = reducers

        log.info("[1] Starting (rank=%d)" % rank)
        self.execute_on_request(self.word_count)
        log.info("[1] Finished (rank=%d)" % rank)
        comm.Barrier()

        log.info("[2] Starting (rank=%d)" % rank)
        self.execute_on_request(self.word_count_per_doc)
        log.info("[2] Finished (rank=%d)" % rank)
        comm.Barrier()

        log.info("[3] Starting (rank=%d)" % rank)
        self.execute_on_request(self.word_scrambler)
        log.info("[3] Finished (rank=%d)" % rank)
        comm.Barrier()

        log.info("[4] Starting (rank=%d)" % rank)
        self.execute_on_request(self.weight)
        log.info("[4] Finished (rank=%d)" % rank)
        comm.Barrier()

    def execute_on_request(self, callback):
        self.tasks = 0
        exit = False

        while not exit:
            comm.send(MSG_STATUS_AVAILABLE, dest=NODE_MASTER)
            msg = comm.recv(source=NODE_MASTER)

            if msg == MSG_COMMAND_QUIT:
                log.info("Terminating mapper on %s (rank=%d) as requested"
                          " after having computed %d tasks." % \
                          (name, rank, self.tasks))

                # We have also to send termination message to the reducers

                for reducer in self.reducers:
                    log.info("Sending termination message to %d" % reducer)
                    comm.ssend(MSG_COMMAND_QUIT, dest=reducer)

                exit = True
            elif msg == MSG_COMMAND_WAIT:
                log.info("Sleeping 2 seconds")
                sleep(2)
            else:
                callback(msg)

    def weight(self, (path, cnt_path, num_docs)):
        self.tasks += 1
        reducers = self.reducers

        with open(path, 'r') as f1:
            with open(cnt_path, 'r') as f2:
                with contextlib.closing(mmap.mmap(f1.fileno(), 0,
                                        access=mmap.ACCESS_READ)) as m1:
                    with contextlib.closing(mmap.mmap(f2.fileno(), 0,
                                            access=mmap.ACCESS_READ)) as m2:

                        line1 = m1.readline()
                        line2 = m2.readline()

                        while line1 and line2:
                            word, doc_id, word_count, word_per_doc = \
                                line1.strip().split(' ', 3)
                            doc_per_word = line2.strip()

                            doc_id = int(doc_id)
                            word_count = int(word_count)
                            word_per_doc = int(word_per_doc)
                            doc_per_word = int(doc_per_word)

                            weight = (float(word_count) /             \
                                      float(word_per_doc)) *          \
                                     math.log(float(num_docs) /       \
                                              float(doc_per_word), 2)

#                            print "%s %d / %d * log(%d / %d)" % (word, word_count,
#                                    word_per_doc, num_docs, doc_per_word)

                            if weight != 0:
                                dst_reducer = hash(doc_id) % len(reducers)
                                comm.send((doc_id, word, weight),
                                          dest=reducers[dst_reducer])

                            line1 = m1.readline()
                            line2 = m2.readline()

        log.info("Removing file %s" % path)
        os.unlink(path)

        log.info("Removing file %s" % cnt_path)
        os.unlink(cnt_path)

    def word_scrambler(self, path):
        self.tasks += 1
        reducers = self.reducers

        with open(path, 'r') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0,
                                    access=mmap.ACCESS_READ)) as m:
                line = m.readline()

                while line:
                    doc_id, word, word_count, word_per_doc = \
                            line.strip().split(' ', 3)

                    dst_reducer = hash(word) % len(reducers)
                    comm.send((word, int(doc_id), int(word_count),
                               int(word_per_doc)), dest=reducers[dst_reducer])
                    line = m.readline()

        log.info("Removing file %s" % path)
        os.unlink(path)

    def word_count_per_doc(self, path):
        self.tasks += 1
        reducers = self.reducers

        # TODO: Here we can optimize the communication by buffering a little
        #       bit instead of sending a message for each word.

        with open(path, 'r') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0,
                                    access=mmap.ACCESS_READ)) as m:
                line = m.readline()

                while line:
                    word, doc_id, word_count = line.strip().split(' ', 2)

                    dst_reducer = hash(word) % len(reducers)
                    comm.send((int(doc_id), word, int(word_count)),
                              dest=reducers[dst_reducer])

                    line = m.readline()

        log.info("Removing file %s" % path)
        os.unlink(path)

    def word_count(self, (path, doc_id)):
        reducers = self.reducers

        # TODO: Here we can optimize the communication by buffering a little
        #       bit instead of sending a message for each word.

        for word in DocumentExtractor(path, doc_id).get_words():
            dst_reducer = hash(word) % len(reducers)
            comm.send((word, doc_id), dest=reducers[dst_reducer])
            self.tasks += 1
