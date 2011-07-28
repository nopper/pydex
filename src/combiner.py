#!/usr/bin/env python
import os
import mmap
import contextlib

from mpi4py import MPI
from heapq import merge, heapify, heappop, heapreplace

from tags import *
from logger import get_logger
from struct import pack, unpack, unpack_from
from tempfile import NamedTemporaryFile

log = get_logger('combiner')

rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD

class KeyValueReader(object):
    def __init__(self, path, converter=None, delete=True):
        """
        @param path the file to read key-values from
        @param converter the converter to apply in order to parse the lines
        @param delete set to True if you want to delete the file after having
                      read all the contents
        """
        self.path = path
        self.delete = delete
        self.converter = converter or KeyValueReader.first_phase

    @staticmethod
    def first_phase(line):
        word, doc_id, counter = line.split(' ', 2)
        doc_id = int(doc_id)
        counter = int(counter)

        return (word, doc_id, counter)

    @staticmethod
    def second_phase(line):
        doc_id, word, word_count, word_per_doc = line.split(' ', 3)
        doc_id = int(doc_id)
        word_count = int(word_count)
        word_per_doc = int(word_per_doc)

        return (doc_id, word, word_count, word_per_doc)

    @staticmethod
    def third_phase(line):
        word, doc_id, word_count, word_per_doc, docs_per_word = line.split(' ', 4)
        doc_id = int(doc_id)
        word_count = int(word_count)
        word_per_doc = int(word_per_doc)
        docs_per_word = int(docs_per_word)

        return (word, doc_id, word_count, word_per_doc, docs_per_word)


    def iterate(self):
        with open(self.path, 'r') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0,
                                    access=mmap.ACCESS_READ)) as m:
                line = m.readline()

                while line:
                    yield (self.converter(line.strip()))
                    line = m.readline()

        if self.delete:
            os.unlink(self.path)

# This shit can be also implemented as a CUDA shitty stuff
class Combiner(object):
    def __init__(self, num_mappers, input_path):
        self.num_mappers = num_mappers
        self.input_path = input_path

        self.is_master_warned = False
        self.input_path = input_path
        self.last_partition = 0

        log.info("[0] Combiner at %s (rank=%d). Blocked on a barrier" % (name, rank))
        comm.Barrier()

        log.info("[2] Starting first phase")
        self.first_phase()

        log.info("[2] Finished combining. Waiting all the reducers for the 2 phase")
        comm.Barrier()

        # The combiner is ahead of one round in fact it process data of the jth
        # -1 stage while mappers are processing results of the jth phase. The
        # pipeline approach is therefore here.

        self.is_master_warned = False
        self.last_partition = 0

        log.info("[3] Starting third phase")
        self.second_phase()
        log.info("[3] Finished")

        comm.Barrier()

        self.is_master_warned = False
        self.last_partition = 0

        log.info("[4] Starting fourth phase")
        self.third_phase()
        log.info("[4] Finished")

    def third_phase(self):
        inputs = []

        for path in os.listdir(self.input_path):
            if not path.startswith("reduce-3-"):
                continue

            path = os.path.join(self.input_path, path)

            inputs.append(KeyValueReader(path,
                                         KeyValueReader.third_phase).iterate())

        heap = []
        for id, iter in enumerate(inputs):
            try:
                item = iter.next()
                toadd = (item, id, iter)
                heap.append(toadd)
            except StopIteration:
                pass

        prev_word = None
        counter = 0
        num_words = 0

        heapify(heap)

        length = 0
        threshold = 1024 * 1024
        sources = set()
        handle, cnt_handle = self.new_assoc_partition()

        while heap:
            item, id, iter = heap[0]

            word, docid, word_count, word_per_doc, docs_per_word = item
            msg = "%s %d %d %d\n" % (word, docid, word_count, word_per_doc)

            if word == prev_word:
                if id not in sources:
                    counter += docs_per_word
                    sources.add(id) # Avoid doble counts

                num_words += 1
            else:
                # Flush information collected so far
                while num_words > 0:
                    cnt_handle.write("%d\n" % counter)
                    num_words -= 1

                counter = docs_per_word
                prev_word = word
                num_words = 1

                sources.clear()
                sources.add(id)

            handle.write(msg)
            length += len(msg)

            try:
                heapreplace(heap, (iter.next(), id, iter))
            except StopIteration:
                heappop(heap)

            if length > threshold:
                length = 0

                while num_words > 0:
                    cnt_handle.write("%d\n" % counter)
                    num_words -= 1

                handle.close()
                cnt_handle.close()

                handle, cnt_handle = self.new_assoc_partition(handle, cnt_handle)


        while num_words > 0:
            cnt_handle.write("%d\n" % counter)
            num_words -= 1

        self.finish_assoc_partition(handle, cnt_handle)

        log.info("Finished combininig the results of the third phase")
        comm.send(MSG_COMMAND_QUIT, dest=NODE_MASTER)

    def second_phase(self):
        inputs = []

        for path in os.listdir(self.input_path):
            if not path.startswith("reduce-2-"):
                continue

            path = os.path.join(self.input_path, path)

            inputs.append(KeyValueReader(path,
                                         KeyValueReader.second_phase).iterate())

        heap = []
        for id, iter in enumerate(inputs):
            try:
                item = iter.next()
                toadd = (item, id, iter)
                heap.append(toadd)
            except StopIteration:
                pass

        prev_docid = -1
        curr_word_per_doc = 0

        docid = -1
        word = None
        word_count = -1

        heapify(heap)

        length = 0
        threshold = 1024 * 1024
        handle = self.new_partition()

        while heap:
            item, id, iter = heap[0]

            if item[0] == docid and item[1] == word:
                word_count += item[2]
            else:
                if word is not None:
                    msg = "%d %s %d %d\n" % \
                        (docid, word, word_count,
                                curr_word_per_doc)

                    handle.write(msg)
                    length += len(msg)

                docid, word, word_count, word_per_doc = item

            if prev_docid != docid:
                curr_word_per_doc = word_per_doc

                for new_item, new_id, new_iter in heap:
                    if new_id != id and new_item[0] == item[0]:
                        curr_word_per_doc += new_item[3]

            prev_docid = docid

            try:
                heapreplace(heap, (iter.next(), id, iter))
            except StopIteration:
                heappop(heap)

            if length > threshold:
                length = 0

                handle.close()
                handle = self.new_partition(handle)

        if word is not None:
            msg = "%d %s %d %d\n" % \
                (docid, word, word_count, curr_word_per_doc)

            handle.write(msg)
            handle.close()

            self.finish_partition(handle)

        log.info("Finished combininig the results of the second phase")
        comm.send(MSG_COMMAND_QUIT, dest=NODE_MASTER)

    def first_phase(self):
        # Ok here we need to open various inputs
        inputs = [os.path.join(self.input_path, path)
            for path in os.listdir(self.input_path)
        ]

        inputs = [KeyValueReader(path).iterate()
            for path in inputs if os.stat(path).st_size != 0
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
        comm.send(MSG_COMMAND_QUIT, dest=NODE_MASTER)


    def finish_partition(self, handle):
        if handle is not None:
            fname = os.path.basename(handle.name)
            os.rename(handle.name,
                      os.path.join(self.input_path, 'input%s' % fname[4:]))

            # Now we can already start the new workers to start the second
            # phase on the previous partition. The idea is to rename the
            # partition file to input-something. The master should be already
            # watching the directory for changes.

            if not self.is_master_warned:
                log.info("Telling the master to start the second phase")

                # It can be whatever since we are not going to check an
                comm.send(0, dest=NODE_MASTER)
                self.is_master_warned = True

    def finish_assoc_partition(self, handle, cnt_handle):
        tmp = self.is_master_warned
        self.is_master_warned = True

        self.finish_partition(handle)

        self.is_master_warned = tmp
        self.finish_partition(cnt_handle)

    def new_partition(self, curr_handle=None):
        handle = \
            NamedTemporaryFile(prefix='part-{:06d}-'.format(self.last_partition),
                               dir=self.input_path, delete=False)
        self.last_partition += 1

        if self.last_partition > 1:
            self.finish_partition(curr_handle)

        return handle

    def new_assoc_partition(self, curr_handle=None, curr_cnt_handle=None):
        handle = \
            NamedTemporaryFile(prefix='part-{:06d}-'.format(self.last_partition),
                               dir=self.input_path, delete=False)
        cnt_handle = \
            NamedTemporaryFile(prefix='part-cnt-{:06d}-'.format(self.last_partition),
                               dir=self.input_path, delete=False)

        self.last_partition += 1

        if self.last_partition > 1:
            self.finish_assoc_partition(curr_handle, curr_cnt_handle)

        return handle, cnt_handle

if __name__ == "__main__":
    Combiner(2, "/home/nopper/Source/pydex/outputs")
