#!/usr/bin/env python

import sys
from tags import *
from mpi4py import MPI
from logger import get_logger
from extractor import DocumentExtractor

log = get_logger('mapper')

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
status = MPI.Status()

class Mapper(object):
    def __init__(self, reducers):
        log.info("Started a new mapper on %s (rank=%d)" % (name, rank))

        self.tasks = 0
        self.reducers = reducers

        self.execute_on_request(self.word_count)
        comm.Barrier()

        #self.execute_on_request()
        #comm.Barrier()

    def execute_on_request(self, callback):
        while True:
            comm.send(None, tag=STATUS_AVAILABLE, dest=0)
            msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == COMMAND_QUIT:
                log.info("Terminating mapper on %s (rank=%d) as requested"
                         " after having computed %d tasks." % \
                         (name, rank, self.tasks))

                # We have also to send termination message to the reducers

                for reducer in self.reducers:
                    comm.send(None, dest=reducer, tag=COMMAND_QUIT)

                break

            callback(msg)

    def word_count(self, (path, doc_id)):
        log.debug("Received a new job '%s'" % (path))

        self.tasks += 1
        reducers = self.reducers
        comm = MPI.COMM_WORLD

        for word in DocumentExtractor(path, doc_id).get_words():
            dst_reducer = hash(word) % len(reducers)
            comm.send((word, doc_id), dest=reducers[dst_reducer])
