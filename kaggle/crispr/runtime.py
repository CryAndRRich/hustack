import os
import sys
import time
import zipfile
WORK = '/tmp/work'
START = time.time()

def out_dir():
    return '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'

def find_data():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = ['/kaggle/input', os.path.join(here, '..', 'data', 'raw'), os.path.join(here, '..', 'data'), '.']
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if 'train.csv' in filenames and 'test.csv' in filenames:
                return dirpath
    raise FileNotFoundError('train.csv and test.csv not found')

class Tee:

    def __init__(self, path):
        self.file = open(path, 'a', buffering=1)
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def log(message):
    print('[%7.1fs] %s' % (time.time() - START, message))

def start_logging():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(WORK, 'extra'), exist_ok=True)
    path = os.path.join(WORK, 'run.log')
    tee = Tee(path)
    sys.stdout = tee
    return tee

def write_results_zip():
    target = os.path.join(out_dir(), 'results.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirnames, filenames in os.walk(WORK):
            for name in filenames:
                full = os.path.join(dirpath, name)
                zf.write(full, os.path.relpath(full, WORK))
    return target
