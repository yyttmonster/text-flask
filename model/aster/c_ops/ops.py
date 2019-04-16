import shutil
import uuid
from os.path import join, dirname, realpath, exists
import tensorflow as tf
import os

tf.app.flags.DEFINE_string('oplib_name', 'aster', 'Name of op library.')
tf.app.flags.DEFINE_string('oplib_suffix', '.so', 'Library suffix.')
FLAGS = tf.app.flags.FLAGS


def _load_oplib(lib_name):
    """
    Load TensorFlow operator library.
    """
    lib_path = join(dirname(realpath(__file__)), 'lib{0}{1}'.format(lib_name, FLAGS.oplib_suffix))
    assert exists(lib_path), '{0} not found'.format(lib_path)
    lib_copy_path = 'model/aster/temporary/lib{0}_{1}{2}'.format(lib_name, str(uuid.uuid4())[:8], FLAGS.oplib_suffix)
    shutil.copyfile(lib_path, lib_copy_path)
    oplib = tf.load_op_library(lib_copy_path)
    return oplib


_oplib = _load_oplib(FLAGS.oplib_name)
string_filtering = _oplib.string_filtering
string_reverse = _oplib.string_reverse
divide_curve = _oplib.divide_curve
