import logging
import time
import datetime
from contextlib import contextmanager
from common import file_operator as f_op
from logging import getLogger, StreamHandler, Formatter, FileHandler

logger_ = None
writer_ = None


@contextmanager
def timer(name, logger_=None, level=logging.DEBUG,
          writer=None, writer_tag=None, writer_gb_step=None,
          collector=None, col_phase=None):
    """
    Logging the time count.
        name: str
            header of log
        logger_: instance of logger
            supposed to set logger only for time count to separate log file
    """
    print_ = print if logger_ is None else lambda msg: logger_.log(level, msg)
    time_st = time.time()
    # print_(f'[{name}] start')
    yield
    cost = time.time() - time_st
    print_(f'[{name}] done in {cost:.0f} s')
    if writer:
        writer.add_scalar(tag=writer_tag, scalar_value=cost, global_step=writer_gb_step)
    if collector:
        collector.set_stat_per_batch(col_phase, "time", cost)


def create_logger(logger_name, log_root_path, save_key, level=logging.INFO):
    # logger object
    logger = getLogger(f"{logger_name}")

    # set debug log level, level is controlled in handler
    logger.setLevel(logging.DEBUG)

    # create log handler
    stream_handler = StreamHandler()
    f_op.create_folder(log_root_path)
    prefix = get_cur_datetime()
    file_handler = FileHandler(f'{log_root_path}/{save_key}_{prefix}.log', 'a')

    # set log level
    stream_handler.setLevel(level)
    file_handler.setLevel(level)

    # set format
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)
    file_handler.setFormatter(handler_format)

    # set handler to logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_log_level_from_name(log_level_name):
    if log_level_name == "DEBUG":
        return logging.DEBUG
    elif log_level_name == "INFO":
        return logging.INFO
    else:
        assert False, f"Log level is either INFO or DEBUG"


def report_dataset_size(train_ds, val_ds):
    train_count = {"0": 0, "1": 0}
    valid_count = {"0": 0, "1": 0}
    train_num_gt_256 = {"0": 0, "1": 0}
    valid_num_gt_256 = {"0": 0, "1": 0}
    for e in train_ds.examples:
        train_count[e.Label] += 1
        train_num_gt_256[e.Label] += 1 if len(e.Text) > 256 else 0
    for e in val_ds.examples:
        valid_count[e.Label] += 1
        valid_num_gt_256[e.Label] += 1 if len(e.Text) > 256 else 0
    logger_.info(f"train num: {len(train_ds)}, neg_num: {train_count['0']}, pos_num: {train_count['1']}, "
                 f"neg_gt_256: {train_num_gt_256['0']}, pos_gt_256: {train_num_gt_256['1']}")
    logger_.info(f"valid num: {len(val_ds)}, neg_num: {valid_count['0']}, pos_num: {valid_count['1']}, "
                 f"neg_gt_256: {valid_num_gt_256['0']}, pos_gt_256: {valid_num_gt_256['1']}")


def get_cur_datetime():
    dt_now = datetime.datetime.now()
    simple_form = dt_now.strftime("%Y%m%d_%H%M%S")
    return simple_form
