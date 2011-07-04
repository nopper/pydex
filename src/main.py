import os
import sys
import numpy

from tags import *
from time import sleep
from logger import get_logger
from mpi4py import MPI

from mapper import Mapper
from reducer import Reducer
from combiner import Combiner

log = get_logger("master")

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

status = MPI.Status()
comm = MPI.COMM_WORLD

SNODES = 2

class Master(object):
    def __init__(self, input_path, num_mappers, num_reducers):
        log.info("Master is located at rank=%d" % rank)
        log.info("Starting the reverse index construction on '%s' with %d" \
                 " mappers and %d reducers." %                             \
                 (input_path, num_mappers, num_reducers))

        self.input_path = input_path
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers

    def start(self):
        self.files = [(os.path.join(self.input_path, path), doc_id)
            for doc_id, path in enumerate(os.listdir(self.input_path))
        ]

        # This is a merely farm. Each worker connects to this server and we
        # assign it a file to scan

        while self.files:
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            if status.Get_tag() == STATUS_AVAILABLE:
                comm.send(self.files.pop(), dest=status.Get_source())

        for i in xrange(self.num_mappers + 1):
            comm.send(None, tag=COMMAND_QUIT,
                      dest=self.num_reducers + i + 1)

        comm.Barrier()
        self.second_phase()

    def refresh_input_list(self, file_set, assigned_set):
        for path in os.listdir(self.input_path):
            fullpath = os.path.join(self.input_path, path)
            if path.startswith("input-") and fullpath not in assigned:
                file_set.add(fullpath)

    def second_phase(self):
        data = comm.recv(source=NODE_COMBINER, tag=COMMAND_START_PHASE2)

        no_more_inputs = False
        assigned = set()
        files = set([os.path.join(self.input_path, path)
            for path in os.listdir(self.input_path)
                if path.startswith("input-")
        ])

        while True:
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            if not no_more_inputs and not files:
                # Refresh our input list in the case the combiner created new
                # inputs

                while True:
                    self.refresh_input_list(files, assigned, True)

                    if not files:
                        log.info("Sleeping 4 seconds for new files to show up")
                        sleep(4)
                    else:
                        break

            # Here we go with a pythonic switch case made of cascaded ifs

            if status.Get_tag() == COMMAND_END_PHASE1:
                self.refresh_input_list(files, assigned)
                no_more_inputs = True

            elif status.Get_tag() == STATUS_AVAILABLE:
                if files:
                    comm.send(files.pop(), dest=status.Get_source())
                else:
                    comm.send(None, dest=status.Get_source(), tag=COMMAND_QUIT)

def initialize_indexer(input_path, output_path, num_mappers, num_reducers):
    if size != 2 + num_mappers + num_reducers:
        print "Error: size does not match"
        sys.exit(-1)
    else:
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)

        if rank == 0:
            master = Master(input_path, num_mappers, num_reducers)
            master.start()
        elif rank == 1:
            Combiner(num_mappers, output_path)

        elif rank < num_reducers + SNODES:
            Reducer(output_path, num_mappers)
        else:
            Mapper([i + SNODES for i in range(num_reducers)])

if __name__ == "__main__":
    if len(sys.argv) != 5:
        if rank == 0:
            print "Usage: %s <input-path> <output-path> <num-mappers> <num-reducers>" % \
                  (sys.argv[0])
        sys.exit(0)
    else:
        initialize_indexer(sys.argv[1], sys.argv[2],
                           int(sys.argv[3]), int(sys.argv[4]))
