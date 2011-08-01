import json

class Configuration(object):
    def __init__(self, fname):
        self.dict = json.load(open(fname, "r"))

    def get(self, string):
        inner = self.dict

        for key in string.split('.'):
            inner = inner[key]

        return inner

conf = Configuration("pydex.json")
