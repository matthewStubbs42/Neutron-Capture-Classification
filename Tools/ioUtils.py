from pathlib import Path
import os

##......................................................................................................##
#                                      Data Read/Write Class                                             #
##......................................................................................................##

class CSVData:

    def __init__(self, dump_path, log_name):
        self.dump_path = dump_path
        self._fout = None
        self._str  = None
        self._dict = {}
        self._path = os.path.join(self.dump_path, log_name)
        #print(os.path.join(self.dump_path, self.log_name))

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]

    def write(self):
        if self._str is None:
            try:
                self._fout=open(self._path, 'w')
            except OSError:
                Path(self.dump_path).mkdir(parents=True, exist_ok=True)
                self._fout=open(self._path, 'w')
#                 print(self._fout)
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))
        self.flush()

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()


