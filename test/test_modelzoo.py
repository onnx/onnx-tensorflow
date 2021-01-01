"""Generates a testing report for ONNX-TF with the ONNX ModelZoo models.

ONNX models found in the ModelZoo directory will be pulled down from
GitHub via `git lfs` (if necessary). The ONNX model will be validated
and converted to a TensorFlow model using ONNX-TensorFlow. A summary
of the conversion will be concatenated into a Markdown-formatted report.

Functions
---------

modelzoo_report(models_dir='models', output_dir=tempfile.gettempdir(),
                include=None, verbose=False, dry_run=False)
"""

import argparse
import datetime
import glob
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import onnx
import tensorflow as tf
import onnx_tf

# Reference matrix on ONNX version, File format version, Opset versions
# https://github.com/onnx/onnx/blob/master/docs/Versioning.md#released-versions

_CFG = {}


class Results:
  """Tracks the detailed status and counts for the report."""

  def __init__(self):
    self.details = []
    self.model_count = 0
    self.total_count = 0
    self.pass_count = 0
    self.warn_count = 0
    self.fail_count = 0
    self.skip_count = 0

  def append_detail(self, line):
    """Append a line of detailed status."""
    self.details.append(line)

  @classmethod
  def _report(cls, line):
    if _CFG['verbose']:
      print(line)
    if not _CFG['dry_run']:
      with open(_CFG['report_filename'], 'a') as file:
        file.write(line + '\n')

  def generate_report(self):
    """Generate the report file."""
    if _CFG['verbose']:
      print('Writing {}{}\n'.format(_CFG['report_filename'],
                                    ' (dry_run)' if _CFG['dry_run'] else ''))
    self._report('*Report generated at {}{}.*'.format(
        datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        _CFG['github_actions_md']))

    self._report('\n## Environment')
    self._report('Package | Version')
    self._report('---- | -----')
    self._report('Platform | {}'.format(platform.platform()))
    self._report('Python | {}'.format(sys.version.replace('\n', ' ')))
    self._report('onnx | {}'.format(onnx.__version__))
    self._report('onnx-tf | {}'.format(_CFG['onnx_tf_version_md']))
    self._report('tensorflow | {}'.format(tf.__version__))

    self._report('\n## Summary')
    self._report('Value | Count')
    self._report('---- | -----')
    self._report('Models | {}'.format(self.model_count))
    self._report('Total | {}'.format(self.total_count))
    self._report(':heavy_check_mark: Passed | {}'.format(self.pass_count))
    self._report(':warning: Limitation | {}'.format(self.warn_count))
    self._report(':x: Failed | {}'.format(self.fail_count))
    self._report(':heavy_minus_sign: Skipped | {}'.format(self.skip_count))

    self._report('\n## Details')
    self._report('\n'.join(self.details))
    self._report('')

  def summary(self):
    """Return the report summary (counts, report location) as a string."""
    return ('Total: {}, Passed: {}, Limitation: {}, Failed: {}, '
            'Skipped: {}\nReport: {}{}').format(
                self.total_count, self.pass_count, self.warn_count,
                self.fail_count, self.skip_count, _CFG['report_filename'],
                ' (dry_run)' if _CFG['dry_run'] else '')


def _get_model_and_test_data():
  """Get the filename of the model and directory of the test data set"""
  onnx_model = None
  test_data_set = []
  for root, dirs, files in os.walk(_CFG['untar_directory']):
    for dir_name in dirs:
      if dir_name.startswith('test_data_set_'):
        test_data_set.append(os.path.join(root, dir_name))
    for file_name in files:
      if file_name.endswith('.onnx') and not file_name.startswith('.'):
        onnx_model = os.path.join(root, file_name)
      elif (file_name.startswith('input_') and file_name.endswith('.pb') and
            len(test_data_set) == 0):
        # data files are not in test_data_set_* but in the same
        # directory of onnx file
        test_data_set.append(root)
      elif file_name.startswith('test_data_') and file_name.endswith('.npz'):
        test_data_file = os.path.join(root, file_name)
        test_data_dir = os.path.join(root, file_name.split('.')[0])
        new_test_data_file = os.path.join(test_data_dir, file_name)
        os.mkdir(test_data_dir)
        os.rename(test_data_file, new_test_data_file)
        test_data_set.append(test_data_dir)
  return onnx_model, test_data_set


def _extract_model_and_test_data(file_path):
  """Extract all files in the tar.gz to test_model_and_data_dir"""
  tar = tarfile.open(file_path, "r:gz")
  tar.extractall(_CFG['untar_directory'])
  tar.close()
  return _get_model_and_test_data()


def _pull_model_file(file_path):
  """Use Git LFS to pull down a large file.

    - If the model file is around ~130B, it's just a file pointer.
      We'll download the file to test, then delete it afterwards
      to minimize disk utilization (important in CI environment).
    - If you previously downloaded the file, the file will remain
      in place after processing. In your local environment, make
      sure to pull the models you test often to avoid repetitive
      downloads.
  """
  model_path = os.path.join(_CFG['models_dir'], file_path)
  file_size = os.stat(model_path).st_size
  pulled = False
  if file_size <= 150:
    # need to pull the model file on-demand using git lfs
    if _CFG['verbose']:
      print('Pulling {}{}'.format(file_path,
                                  ' (dry_run)' if _CFG['dry_run'] else ''))
    if not _CFG['dry_run']:
      cmd_args = 'git lfs pull -I {} -X ""'.format(file_path)
      subprocess.run(cmd_args,
                     cwd=_CFG['models_dir'],
                     shell=True,
                     check=True,
                     stdout=subprocess.DEVNULL)
      if file_path.endswith('.tar.gz'):
        onnx_model, test_data_set = _extract_model_and_test_data(model_path)
      else:
        onnx_model = model_path
        test_data_set = []
      new_size = os.stat(onnx_model).st_size
      pulled = new_size != file_size
      file_size = new_size
  else:
    # model file is pulled already
    if file_path.endswith('.tar.gz'):
      onnx_model, test_data_set = _extract_model_and_test_data(model_path)
    else:
      onnx_model = model_path
      test_data_set = []
  return (file_size, pulled), onnx_model, test_data_set


def _revert_model_pointer(file_path):
  """Remove downloaded model, revert to pointer, remove cached file."""
  cmd_args = ('rm -f {0} && '
              'git reset HEAD {0} && '
              'git checkout {0} && '
              'rm -f $(find . | grep $(grep oid {0} | cut -d ":" -f 2))'
             ).format(file_path)
  subprocess.run(cmd_args,
                 cwd=_CFG['models_dir'],
                 shell=True,
                 check=True,
                 stdout=subprocess.DEVNULL)


def _include_model(file_path):
  if _CFG['include'] is None:
    return True
  for item in _CFG['include']:
    if (file_path.startswith(item) or file_path.endswith(item + '.onnx') or
        '/{}/model/'.format(item) in file_path or
        '/{}/models/'.format(item) in file_path):
      return True
  return False


def _has_models(dir_path):
  for m_dir in ['model', 'models']:
    # in age_gender there are 2 different models in there so the
    # directory is "models" instead of "model" like the rest of
    # the other models
    model_dir = os.path.join(_CFG['models_dir'], dir_path, m_dir)
    if os.path.exists(model_dir):
      for item in os.listdir(model_dir):
        if item.endswith('.onnx'):
          file_path = os.path.join(dir_path, model_dir, item)
          if _include_model(file_path):
            return True
  return False


def _del_location(loc):
  if not _CFG['dry_run'] and os.path.exists(loc):
    if os.path.isdir(loc):
      shutil.rmtree(loc)
    else:
      os.remove(loc)


def _size_with_units(size):
  if size < 1024:
    units = '{}B'.format(size)
  elif size < math.pow(1024, 2):
    units = '{}K'.format(round(size / 1024))
  elif size < math.pow(1024, 3):
    units = '{}M'.format(round(size / math.pow(1024, 2)))
  else:
    units = '{}G'.format(round(size / math.pow(1024, 3)))
  return units


def _report_check_model(model):
  """Use ONNX checker to test if model is valid and return a report string."""
  try:
    onnx.checker.check_model(model)
    return ''
  except Exception as ex:
    _del_location(_CFG['untar_directory'])
    first_line = str(ex).strip().split('\n')[0].strip()
    return '{}: {}'.format(type(ex).__name__, first_line)


def _report_convert_model(model):
  """Test conversion and returns a report string."""
  try:
    tf_rep = onnx_tf.backend.prepare(model)
    tf_rep.export_graph(_CFG['output_directory'])
    return ''
  except Exception as ex:
    _del_location(_CFG['untar_directory'])
    _del_location(_CFG['output_directory'])
    stack_trace = str(ex).strip().split('\n')
    if len(stack_trace) > 1:
      err_msg = stack_trace[-1].strip()
      # OpUnsupportedException gets raised as a RuntimeError
      if 'OP_UNSUPPORTED_EXCEPT' in str(ex):
        err_msg = err_msg.replace(type(ex).__name__, 'OpUnsupportedException')
      return err_msg
    return '{}: {}'.format(type(ex).__name__, stack_trace[0].strip())


def _get_inputs_outputs_pb(tf_rep, data_dir):
  """Get the input and reference output tensors"""
  inputs = {}
  inputs_num = len(glob.glob(os.path.join(data_dir, 'input_*.pb')))
  for i in range(inputs_num):
    input_file = os.path.join(data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
      tensor.ParseFromString(f.read())
      tensor.name = tensor.name if tensor.name else tf_rep.inputs[i]
    inputs[tensor.name] = onnx.numpy_helper.to_array(tensor)
  ref_outputs = {}
  ref_outputs_num = len(glob.glob(os.path.join(data_dir, 'output_*.pb')))
  for i in range(ref_outputs_num):
    output_file = os.path.join(data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
      tensor.ParseFromString(f.read())
      tensor.name = tensor.name if tensor.name else tf_rep.outputs[i]
    ref_outputs[tensor.name] = onnx.numpy_helper.to_array(tensor)
  return inputs, ref_outputs


def _get_inputs_outputs_npz(tf_rep, data_dir):
  """Get the input and reference output tensors"""
  npz_file = os.path.join(data_dir, '{}.npz'.format(data_dir.split('/')[-1]))
  data = np.load(npz_file, encoding='bytes')
  inputs = {}
  ref_outputs = {}
  for i in range(len(tf_rep.inputs)):
    inputs[tf_rep.inputs[i]] = data['inputs'][i]
  for i in range(len(tf_rep.outputs)):
    ref_outputs[tf_rep.outputs[i]] = data['outputs'][i]
  return inputs, ref_outputs


def _get_inputs_and_ref_outputs(tf_rep, data_dir):
  """Get the input and reference output tensors"""
  if len(glob.glob(os.path.join(data_dir, 'input_*.pb'))) > 0:
    inputs, ref_outputs = _get_inputs_outputs_pb(tf_rep, data_dir)
  else:
    inputs, ref_outputs = _get_inputs_outputs_npz(tf_rep, data_dir)
  return inputs, ref_outputs


def _assert_outputs(outputs, ref_outputs, rtol, atol):
  np.testing.assert_equal(len(outputs), len(ref_outputs))
  for key in outputs.keys():
    np.testing.assert_equal(outputs[key].dtype, ref_outputs[key].dtype)
    if ref_outputs[key].dtype == np.object:
      np.testing.assert_array_equal(outputs[key], ref_outputs[key])
    else:
      np.testing.assert_allclose(outputs[key],
                                 ref_outputs[key],
                                 rtol=rtol,
                                 atol=atol)


def _report_run_model(model, data_set):
  """Run the model and returns a report string."""
  try:
    tf_rep = onnx_tf.backend.prepare(model)
    for data in data_set:
      inputs, ref_outputs = _get_inputs_and_ref_outputs(tf_rep, data)
      outputs = tf_rep.run(inputs)
      outputs_dict = {}
      for i in range(len(tf_rep.outputs)):
        outputs_dict[tf_rep.outputs[i]] = outputs[i]
      _assert_outputs(outputs_dict, ref_outputs, rtol=1e-3, atol=1e-3)
  except Exception as ex:
    stack_trace = str(ex).strip().split('\n')
    if len(stack_trace) > 1:
      if ex.__class__ == AssertionError:
        return stack_trace[:5]
      else:
        return stack_trace[-1].strip()
    return '{}: {}'.format(type(ex).__name__, stack_trace[0].strip())
  finally:
    _del_location(_CFG['untar_directory'])
    _del_location(_CFG['output_directory'])


def _report_model(file_path, results=Results(), onnx_model_count=1):
  """Generate a report status for a single model, and append it to results."""
  size_pulled, onnx_model, test_data_set = _pull_model_file(file_path)
  if _CFG['dry_run']:
    ir_version = ''
    opset_version = ''
    check_err = ''
    convert_err = ''
    ran_err = ''
    emoji_validated = ''
    emoji_converted = ''
    emoji_ran = ''
    emoji_overall = ':heavy_minus_sign:'
    results.skip_count += 1
  else:
    if _CFG['verbose']:
      print('Testing', file_path)
    model = onnx.load(onnx_model)
    ir_version = model.ir_version
    opset_version = model.opset_import[0].version
    check_err = _report_check_model(model)
    convert_err = '' if check_err else _report_convert_model(model)
    run_err = '' if convert_err or len(
        test_data_set) == 0 else _report_run_model(model, test_data_set)

    if (not check_err and not convert_err and not run_err and
        len(test_data_set) > 0):
      # https://github-emoji-list.herokuapp.com/
      # ran successfully
      emoji_validated = ':ok:'
      emoji_converted = ':ok:'
      emoji_ran = ':ok:'
      emoji_overall = ':heavy_check_mark:'
      results.pass_count += 1
    elif (not check_err and not convert_err and not run_err and
          len(test_data_set) == 0):
      # validation & conversion passed but no test data available
      emoji_validated = ':ok:'
      emoji_converted = ':ok:'
      emoji_ran = 'No test data provided in model zoo'
      emoji_overall = ':warning:'
      results.warn_count += 1
    elif not check_err and not convert_err:
      # validation & conversion passed but failed to run
      emoji_validated = ':ok:'
      emoji_converted = ':ok:'
      emoji_ran = run_err
      emoji_overall = ':x:'
      results.fail_count += 1
    elif not check_err:
      # validation pass, but conversion did not
      emoji_validated = ':ok:'
      emoji_converted = convert_err
      emoji_ran = ':heavy_minus_sign:'
      if ('BackendIsNotSupposedToImplementIt' in convert_err or
          'OpUnsupportedException' in convert_err):
        # known limitations
        # - BackendIsNotSupposedToImplementIt: Op not implemented
        # - OpUnsupportedException: TensorFlow limitation
        emoji_overall = ':warning:'
        results.warn_count += 1
      else:
        # conversion failed
        emoji_overall = ':x:'
        results.fail_count += 1
    else:
      # validation failed
      emoji_validated = check_err
      emoji_converted = ':heavy_minus_sign:'
      emoji_ran = ':heavy_minus_sign:'
      emoji_overall = ':x:'
      results.fail_count += 1

  results.append_detail('{} | {}. {} | {} | {} | {} | {} | {} | {}'.format(
      emoji_overall, onnx_model_count,
      file_path[file_path.rindex('/') + 1:file_path.index('.')],
      _size_with_units(size_pulled[0]), ir_version, opset_version,
      emoji_validated, emoji_converted, emoji_ran))

  if len(test_data_set) == 0:
    _del_location(_CFG['output_directory'])
  if size_pulled[1]:
    # only remove model if it was pulled above on-demand
    _revert_model_pointer(file_path)


def _configure(models_dir='models',
               output_dir=tempfile.gettempdir(),
               include=None,
               verbose=False,
               dry_run=False):
  """Validate the configuration."""
  if not os.path.isdir(models_dir):
    raise NotADirectoryError(models_dir)
  if not os.path.isdir(output_dir):
    raise NotADirectoryError(output_dir)
  subprocess.run('git lfs', shell=True, check=True, stdout=subprocess.DEVNULL)

  _CFG['models_dir'] = os.path.normpath(models_dir)
  _CFG['include'] = include.split(',') \
      if isinstance(include, str) else include
  _CFG['verbose'] = verbose
  _CFG['dry_run'] = dry_run

  _configure_env()

  norm_output_dir = os.path.normpath(output_dir)
  _CFG['untar_directory'] = os.path.join(norm_output_dir, 'test_model_and_data')
  _CFG['output_directory'] = os.path.join(norm_output_dir, 'test_model_pb')
  _CFG['report_filename'] = os.path.join(norm_output_dir,
                                         _CFG['report_filename'])


def _configure_env():
  """Set additional configuration based on environment variables."""
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  ref = os.getenv('GITHUB_REF')
  repo = os.getenv('GITHUB_REPOSITORY')
  sha = os.getenv('GITHUB_SHA')
  run_id = os.getenv('GITHUB_RUN_ID')

  if ref and '/' in ref:
    ref_type = 'tag' if '/tags/' in ref else 'branch'
    ref_name = ref[str(ref).rindex('/') + 1:]
    report_md = 'ModelZoo-Status-({}={}).md'.format(ref_type, ref_name)
  else:
    report_md = 'ModelZoo-Status.md'
  _CFG['report_filename'] = report_md

  if repo:
    # actions ([run_id](url))
    actions_url = 'https://github.com/{}/actions'.format(repo)
    _CFG['github_actions_md'] = ' via [GitHub Actions]({})'.format(actions_url)
    if run_id:
      run_link = ' ([{0}]({1}/runs/{0}))'.format(run_id, actions_url)
      _CFG['github_actions_md'] += run_link
  else:
    _CFG['github_actions_md'] = ''

  _CFG['onnx_tf_version_md'] = onnx_tf.version.version
  if sha and repo:
    # version ([sha](url))
    commit_url = 'https://github.com/{}/commit/{}'.format(repo, sha)
    _CFG['onnx_tf_version_md'] += ' ([{}]({}))'.format(sha[0:7], commit_url)


def modelzoo_report(models_dir='models',
                    output_dir=tempfile.gettempdir(),
                    include=None,
                    verbose=False,
                    dry_run=False):
  """Generate a testing report for the models found in the given directory.

    ONNX models found in the ModelZoo directory will be pulled down from
    GitHub via `git lfs` (if necessary). The ONNX model will be validated
    and converted to a TensorFlow model using ONNX-TensorFlow. A summary
    of the conversion will be concatenated into a Markdown-formatted report.

    Args:
        models_dir: directory that contains ONNX models
        output_dir: directory for the generated report and converted model
        include: comma-separated list of models or paths to include
        verbose: verbose output
        dry_run: process directory without doing conversion

    Returns:
        Results object containing detailed status and counts for the report.
    """

  _configure(models_dir, output_dir, include, verbose, dry_run)
  _del_location(_CFG['report_filename'])
  _del_location(_CFG['output_directory'])
  _del_location(_CFG['untar_directory'])

  # run tests first, but append to report after summary
  results = Results()
  for root, subdir, files in os.walk(_CFG['models_dir']):
    subdir.sort()
    if 'model' in subdir or 'models' in subdir:
      dir_path = os.path.relpath(root, _CFG['models_dir'])
      if _has_models(dir_path):
        results.model_count += 1
        results.append_detail('')
        results.append_detail('### {}. {}'.format(results.model_count,
                                                  os.path.basename(root)))
        results.append_detail(dir_path)
        results.append_detail('')
        results.append_detail(
            'Status | Model | Size | IR | Opset | ONNX Checker | '
            'ONNX-TF Converted | ONNX-TF Ran')
        results.append_detail(
            '------ | ----- | ---- | -- | ----- | ------------ | '
            '----------------- | -----------')
    onnx_model_count = 0
    file_path = ''
    for item in sorted(files):
      if item.endswith('.onnx'):
        file_path = os.path.relpath(os.path.join(root, item),
                                    _CFG['models_dir'])
        # look for gz file for this model
        gzfile_path = file_path.replace('.onnx', '.tar.gz')
        gzfile_path = os.path.join(_CFG['models_dir'], gzfile_path)
        if gzfile_path in glob.glob(gzfile_path):
          file_path = os.path.relpath(gzfile_path, _CFG['models_dir'])
        if _include_model(file_path):
          onnx_model_count += 1
          results.total_count += 1
          _report_model(file_path, results, onnx_model_count)

  return results


if __name__ == '__main__':
  tempdir = tempfile.gettempdir()
  parser = argparse.ArgumentParser(
      description=('Test converting ONNX ModelZoo models to TensorFlow. '
                   'Prerequisite: `git lfs`'))
  parser.add_argument('-m',
                      '--models',
                      default='models',
                      help=('ONNX ModelZoo directory (default: models)'))
  parser.add_argument('-o',
                      '--output',
                      default=tempdir,
                      help=('output directory (default: {})'.format(tempdir)))
  parser.add_argument(
      '-i',
      '--include',
      help=('comma-separated list of models or paths to include. '
            'Use `git lfs pull` to cache frequently tested models.'))
  parser.add_argument('-v',
                      '--verbose',
                      action='store_true',
                      help=('verbose output'))
  parser.add_argument('--dry-run',
                      action='store_true',
                      help=('process directory without doing conversion'))
  args = parser.parse_args()
  report = modelzoo_report(args.models, args.output, args.include, args.verbose,
                           args.dry_run)
  report.generate_report()
  print(report.summary())
