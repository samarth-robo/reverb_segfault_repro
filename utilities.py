import logging
import sys
import numpy as np
import os.path as osp
import time


if 'absl.logging' in sys.modules:
  import absl.logging
  absl.logging.set_verbosity('info')
  absl.logging.set_stderrthreshold('info')


def setup_logging(filename=None, level=logging.INFO):
  handlers = []
  handlers.append(logging.StreamHandler(sys.stdout))
  if filename is not None:
    filename = osp.expanduser(filename)
    handlers.append(logging.FileHandler(filename, 'w'))

  logging.basicConfig(level=level, handlers=handlers)

  if filename is not None:
    root = logging.getLogger()
    root.info('Logging to {:s}'.format(filename))


def log_dict(logger: logging.Logger, d: dict, prefix=''):
  for k, v in d.items():
    if isinstance(v, dict):
      logger.info(f'{prefix}{k}:')
      log_dict(logger, v, prefix=f'{prefix}\t')
    else:
      logger.info(f'{prefix}{k}: {v}')


class DuplicateLogFilter:
    """
    Filters away duplicate log messages.
    Modified version of: https://stackoverflow.com/a/31953563/965332
    """

    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record: logging.LogRecord):
        msg = str(record.msg % record.args)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeFilter(self)


class RunningMean:
  def __init__(self):
    self._x = 0
    self.count = 0

  def update(self, x):
    self._x += x
    self.count += 1

  def get_value(self):
    return self._x / self.count


class StopWatch:
  def __init__(self, avg_mode=False):
    self.x = 0
    self.start_time = 0
    self.avg_mode = avg_mode

  def start(self, x):
    self.x = x
    self.start_time = time.time()

  def stop(self, x, override_time=None):
    time_spent = override_time or (time.time() - self.start_time)
    work_done = np.mean(x - self.x) if self.avg_mode else np.sum(x - self.x)
    speed = work_done / time_spent
    return speed, time_spent
