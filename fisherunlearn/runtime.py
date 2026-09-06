DEVICE = "cpu"
INFO_BATCH_SIZE = 1
MIA_BATCH_SIZE = 1


def set_device(device):
    global DEVICE
    DEVICE = device


def set_info_batch_size(batch_size):
    global INFO_BATCH_SIZE
    INFO_BATCH_SIZE = batch_size


def set_mia_batch_size(batch_size):
    global MIA_BATCH_SIZE
    MIA_BATCH_SIZE = batch_size
