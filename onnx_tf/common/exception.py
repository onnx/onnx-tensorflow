import inspect
import warnings


class CustomException(object):

  def __init__(self):
    self._func = RuntimeError
    self._message = ""

  def __call__(self, *args, **kwargs):
    if inspect.isclass(self._func) and issubclass(self._func, Exception):
      raise self._func(self.get_message(*args, **kwargs))
    elif callable(self._func):
      self._func(self.get_message(*args, **kwargs))

  def get_message(self, *args, **kwargs):
    return self._message


class OpUnimplementedException(CustomException):

  def __init__(self):
    super(OpUnimplementedException, self).__init__()
    self._func = NotImplementedError
    self._message = "{} is not implemented."

  def __call__(self, op, version=None):
    if IGNORE_UNIMPLEMENTED:
      self._func = warnings.warn
    super(OpUnimplementedException, self).__call__(op, version)

  def get_message(self, op, version=None):
    insert_message = op
    if version is not None:
      insert_message = "{} with version of {}".format(op, version)
    return self._message.format(insert_message)


class OpUnsupportedException(object):

  def __init__(self):
    super(OpUnsupportedException, self).__init__()
    self._func = RuntimeError
    self._message = "{} is not supported in {}."

  def __call__(self, op, framework):
    raise self._func(self.get_message(op, framework))

  def get_message(self, op, framework):
    return self._message.format(op, framework)


class ConstNotFoundException(CustomException):

  def __init__(self):
    super(ConstNotFoundException, self).__init__()
    self._func = RuntimeError
    self._message = "{} of {} is not found in graph consts."

  def __call__(self, name, op):
    super(ConstNotFoundException, self).__call__(name, op)

  def get_message(self, name, op):
    return self._message.format(name, op)


IGNORE_UNIMPLEMENTED = False
OP_UNIMPLEMENTED_EXCEPT = OpUnimplementedException()
OP_UNSUPPORTED_EXCEPT = OpUnsupportedException()
CONST_NOT_FOUND_EXCEPT = ConstNotFoundException()
