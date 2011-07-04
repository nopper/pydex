import mmap
import contextlib
from parsing import read_file, preprocess_string

class DocumentExtractor(object):
    def __init__(self, path, doc_id):
        self.path = path
        self.doc_id = doc_id

    def get_words(self):
        with open(self.path, 'r') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), 0,
                                    access=mmap.ACCESS_READ)) as m:

                for line in f.readlines():
                    for word in preprocess_string(line):
                        yield word
