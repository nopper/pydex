#!/usr/bin/env python
import os
import mmap
import contextlib

from mpi4py import MPI
from heapq import merge

from tags import *
from logger import get_logger
from struct import pack, unpack, unpack_from
from tempfile import NamedTemporaryFile

log = get_logger('combiner')

rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

class KeyValueReader(object):
    def __init__(self, path, delete=True):
        """
        @param path the file to read key-values from
        @param delete set to True if you want to delete the file after having
                      read all the contents
        """
        self.path = path
        self.delete = delete

    def iterate(self):
        with open(self.path, 'r') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0,
                                    access=mmap.ACCESS_READ)) as m:
                for line in f.readlines():
                    word, doc_id, counter = line.strip().split(' ', 2)
                    doc_id = int(doc_id)
                    counter = int(counter)

                    yield (word, doc_id, counter)

        if self.delete:
            os.unlink(self.path)

# This shit can be also implemented as a CUDA shitty stuff
class Combiner(object):
    def __init__(self, num_mappers, input_path):
        log.info("Started a new combiner on %s (rank=%d). Blocked on a barrier" % (name, rank))
        comm.Barrier()

        self.input_path = input_path
        self.last_partition = 0

        # Ok here we need to open various inputs
        inputs = [
            KeyValueReader(os.path.join(input_path, path)).iterate()
                for path in os.listdir(input_path)
        ]

        length = 0
        threshold = 1024 * 1024

        handle = self.new_partition()
        last_word, last_doc_id, last_count = '', 0, 0

        # I don't like this style though
        for item in merge(*inputs):
            word, doc_id, count = item

            if doc_id == last_doc_id and word == last_word:
                last_count += count
            else:
                if last_count > 0:
                    msg = "%s %d %d\n" % (last_word, last_doc_id, last_count)
                    handle.write(msg)
                    length += len(msg)

                last_count = count

            # Here we should have contiguous partition so a check on the latest
            # word is also required

            if length > threshold and last_count == 0:
                length = 0

                handle.close()
                handle = self.new_partition(handle)

            last_word, last_doc_id = word, doc_id

        # Here we close the pending handle and signal to the master that the
        # first phase has been concluded

        if last_count > 0:
            handle.write("%s %d %d\n" % \
                         (last_word, last_doc_id, last_count))

        handle.close()
        self.finish_partition(handle)

        log.info("Finished combininig the results of the first phase")

        MPI.COMM_WORLD.send(None, dest=NODE_MASTER, tag=COMMAND_END_PHASE1)

    def finish_partition(self, handle):
        if handle is not None:
            fname = os.path.basename(handle.name)
            os.rename(handle.name,
                      os.path.join(self.input_path, 'input%s' % fname[4:]))

    def new_partition(self, curr_handle=None):
        handle = \
            NamedTemporaryFile(prefix='part-{:06d}-'.format(self.last_partition),
                               dir=self.input_path, delete=False)
        self.last_partition += 1

        if self.last_partition > 1:
            # Now we can already start the new workers to start the second
            # phase on the previous partition. The idea is to rename the
            # partition file to input-something. The master should be already
            # watching the directory for changes.

            self.finish_partition(curr_handle)

            if not self.is_master_warned:
                log.info("Telling the master to start the second phase")
                MPI.COMM_WORLD.send(None, dest=0, tag=COMMAND_PHASE2)
                self.is_master_warned = True

        return handle
