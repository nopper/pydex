#!/usr/bin/env python

from mpi4py import MPI
from heapq import heappush, heappop
from tempfile import NamedTemporaryFile

from struct import calcsize, pack
from tags import *
from logger import get_logger

log = get_logger('reducer')

rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

class Reducer(object):
    def __init__(self, output_path, num_mappers):
        log.info("Started a new reducer on %s (rank=%d)" % (name, rank))

        self.output_path = output_path
        self.num_workers = num_mappers

        status = MPI.Status()
        comm = MPI.COMM_WORLD

        heap = []
        length = 0
        threshold = 1024 * 1024 # 1 MByte
        remaining = num_mappers

        while remaining > 0:
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == COMMAND_QUIT:
                remaining -= 1
            else:
                word, doc_id = msg

                length += len(str(doc_id)) + len(word) + 1
                heappush(heap, msg)

            if length > threshold or remaining == 0:
                self.write_partition(heap)
                heap = []
                length = 0

        comm.Barrier()

        log.info("Ready for the second phase")

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
