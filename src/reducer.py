#!/usr/bin/env python

from mpi4py import MPI
from logger import get_logger

log = get_logger('reducer')

rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

class Reducer(object):
    def __init__(self):
        log.info("Started a new reducer on %s (rank=%d)" % (name, rank))

if __name__ == "__main__":
    Reducer()
