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
    def __init__(self, input_path, output_path, num_mappers, num_reducers):
        log.info("[0] Master is located at rank=%d" % rank)
        log.info("[0] Starting the reverse index construction on '%s' " \
                 "with %d mappers and %d reducers." %                   \
                 (input_path, num_mappers, num_reducers))

        self.num_docs = 0
        self.input_path = input_path
        self.output_path = output_path
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers

    def start(self):
        log.info("[1] Starting")
        self.first_phase()
        log.info("[1] Finished. Waiting on a Barrier")
        comm.Barrier()

        log.info("[2] Starting")
        self.second_phase()
        log.info("[2] Finished. Waiting on a Barrier")
        comm.Barrier()

        log.info("[3] Starting")
        self.third_phase()
        log.info("[3] Finished. Waiting on a Barrier")
        comm.Barrier()

    def third_phase(self):
        self.second_phase()

    def first_phase(self):
        docid = 1
        self.files = []

        for path in sorted(os.listdir(self.input_path)):
            path = os.path.join(self.input_path, path)

            if os.stat(path).st_size == 0:
                continue

            self.files.append((path, docid))
            docid += 1

        self.num_docs = docid - 1

        # This is a merely farm. Each worker connects to this server and we
        # assign it a file to scan

        while self.files:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)

            if msg == MSG_STATUS_AVAILABLE:
                comm.send(self.files.pop(), dest=status.Get_source())

        for i in xrange(self.num_mappers):
            log.debug("Sending termination message to %d" % \
                      (self.num_reducers + i + SNODES))
            comm.send(MSG_COMMAND_QUIT, dest=self.num_reducers + i + SNODES)

    def refresh_input_list(self, file_set, assigned_set):
        for path in os.listdir(self.output_path):
            if path.startswith("input-") and path not in assigned_set:
                file_set.add(path)

        return file_set

    def second_phase(self):
        log.debug("Waiting for message from the combiner. About to start")
        data = comm.recv(source=NODE_COMBINER)
        log.debug("Received. Entering the second phase loop")

        no_more_inputs = False
        assigned = set()
        files = set()

        mappers = set()

        while len(mappers) != self.num_mappers:
            msg = comm.recv(source=MPI.ANY_SOURCE, status=status)
            source = status.Get_source()

            if not no_more_inputs and not files:
                files = self.refresh_input_list(files, assigned)

            # Here we go with a pythonic switch case made of cascaded ifs

            if source == NODE_COMBINER and msg == MSG_COMMAND_QUIT:
                files = self.refresh_input_list(files, assigned)
                no_more_inputs = True

                log.debug("Combiner has finished the first phase. No more "
                          "inputs are available (%d left)" % len(files))

            elif msg == MSG_STATUS_AVAILABLE:
                if files:
                    target = files.pop()
                    assigned.add(target)

                    comm.send(os.path.join(self.output_path, target),
                              dest=source)
                else:
                    if no_more_inputs and source not in mappers:
                        log.info("Sending termination messages for the second "
                                  "phase to mapper to %d" % source)
                        comm.send(MSG_COMMAND_QUIT, dest=source)
                        mappers.add(source)
                    else:
                        comm.send(MSG_COMMAND_WAIT, dest=source)

def initialize_indexer(input_path, output_path, num_mappers, num_reducers):
    if size != 2 + num_mappers + num_reducers:
        print "Error: size does not match"
        sys.exit(-1)
    else:
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)

        if rank == 0:
            master = Master(input_path, output_path, num_mappers, num_reducers)
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
