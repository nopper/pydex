import mmap
import tarfile
import contextlib
from parsing import read_file, preprocess_string

class DocumentExtractor(object):
    def __init__(self, path):
        self.path = path

    def get_words(self):
        with open(self.path, 'r') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0,
                                    access=mmap.ACCESS_READ)) as m:
                with tarfile.open(mode='r:gz', fileobj=m) as archive:
                    member = archive.next()

                    while member != None:
                        doc_id = int(member.name.split("-", 2)[1])
                        handle = archive.extractfile(member)
                        line = handle.readline()

                        perc = float(m.tell()) / float(m.size())

                        while handle.tell() < handle.size:
                            for word in preprocess_string(line):
                                yield (perc, doc_id, word)
                            line = handle.readline()

                        member = archive.next()
