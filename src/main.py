import os
import sys
import numpy

from tags import *
from logger import get_logger
from mpi4py import MPI

log = get_logger("master")

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

class Master(object):
    def __init__(self, input_path, num_mappers, num_reducers):
        log.info("Starting the reverse index construction on '%s' with %d" \
                 " mappers and %d reducers." %                             \
                 (input_path, num_mappers, num_reducers))

        self.input_path = input_path

        self.comm = MPI.COMM_WORLD.Spawn_multiple(
            [sys.executable, sys.executable],
            args=[['mapper.py'], ['reducer.py']],
            maxprocs=[num_mappers, num_reducers]
        )

    def start(self):
        self.files = [os.path.join(self.input_path, path)
            for path in os.listdir(self.input_path)
        ]

        # This is a merely farm. Each worker connects to this server and we
        # assign it a file to scan

        status = MPI.Status()
        comm = self.comm

        while self.files:
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == STATUS_AVAILABLE:
                comm.send(self.files.pop(), dest=status.Get_source())

        for i in xrange(comm.size + 1):
            comm.send(None, tag=COMMAND_QUIT, dest=i)

def start_indexer(input_path, num_mapper, num_reducer):
    master = Master(os.path.abspath(input_path), num_mapper, num_reducer)
    master.start()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Usage: %s <input-path> <num-mappers> <num-reducers>" % \
              (sys.argv[0])
        sys.exit(0)
    else:
        start_indexer(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
