from tensorflow.contrib.grpc.ops.gen_grpc_ops import *
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_grpc_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_grpc_ops.so")
)
