#!/usr/bin/env python

from tags import *
from mpi4py import MPI
from logger import get_logger

log = get_logger('mapper')

rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.Comm.Get_parent()

class Mapper(object):
    def __init__(self):
        log.info("Started a new mapper on %s (rank=%d)" % (name, rank))

        self.tasks = 0
        status = MPI.Status()

        while True:
            comm.send(None, tag=STATUS_AVAILABLE, dest=0)
            msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == COMMAND_QUIT:
                log.info("Terminating mapper on %s (rank=%d) as requested"
                         " after having computed %d tasks." % \
                         (name, rank, self.tasks))
                break

            self.run_job(msg)

    def run_job(self, path):
        log.debug("Received a new job '%s'" % (path))
        self.tasks += 1

if __name__ == "__main__":
    Mapper()
