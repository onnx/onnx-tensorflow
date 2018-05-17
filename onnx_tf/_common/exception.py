import warnings


class CustomException:

  def __init__(self):
    self._error = RuntimeError
    self._message = ""

  def __call__(self, *args, **kwargs):
    if USE_WARNING:
      warnings.warn(self._message)
    else:
      raise self._error(self._message)

  def get_message(self, *args, **kwargs):
    return self._message


class OpNotImplementedException(CustomException):

  def __init__(self):
    super(OpNotImplementedException, self).__init__()
    self._error = NotImplementedError
    self._message = "{} is not implemented."

  def __call__(self, op):
    super(OpNotImplementedException, self).__call__(op)

  def get_message(self, op):
    self._message.format(op)


class OpNotSupportedException(CustomException):

  def __init__(self):
    super(OpNotSupportedException, self).__init__()
    self._error = RuntimeError
    self._message = "{} is not supported in {}."

  def __call__(self, op, framework):
    super(OpNotSupportedException, self).__call__(op, framework)

  def get_message(self, op, framework):
    self._message.format(op, framework)


USE_WARNING = False
OP_NOT_IMPL_EXCEPT = OpNotImplementedException()
OP_NOT_SUPPORT_EXCEPT = OpNotSupportedException()
