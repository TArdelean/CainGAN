import numbers
import argparse


class ParsableOptions:
    def __init__(self, suppress_parse=False):
        self.initialize()
        if not suppress_parse:
            self.parse()
            self.proc()

    def initialize(self):
        """ Method were all fields should be initialized. Variables starting with _ will not be parsed"""
        pass

    def proc(self):
        """ Post processing, after the options have been parsed """
        pass

    @staticmethod
    def good_instance(val):
        return isinstance(val, str) or (isinstance(val, numbers.Number) and not isinstance(val, bool))

    def parse(self):
        parser = argparse.ArgumentParser()
        for name, val in vars(self).items():
            if name.startswith("_"):
                continue
            like = type(val) if ParsableOptions.good_instance(val) else eval
            parser.add_argument(f'--{name}', type=like, default=val, help="It is obvious")

        args = parser.parse_args()
        for name, val in vars(self).items():
            if name.startswith("_"):
                continue
            attr = getattr(args, name)
            self.__setattr__(name, attr)
