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
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile

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
      new_size = os.stat(model_path).st_size
      pulled = new_size != file_size
      file_size = new_size
  return (file_size, pulled)


def _revert_model_pointer(file_path):
  """Remove downloaded model, revert to pointer, remove cached file."""
  cmd_args = ('rm -f {0} && '
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
        '/{}/model/'.format(item) in file_path):
      return True
  return False


def _has_models(dir_path):
  for item in os.listdir(os.path.join(_CFG['models_dir'], dir_path, 'model')):
    if item.endswith('.onnx'):
      file_path = os.path.join(dir_path, 'model', item)
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
    first_line = str(ex).strip().split('\n')[0].strip()
    return '{}: {}'.format(type(ex).__name__, first_line)


def _report_convert_model(model):
  """Test conversion and returns a report string."""
  try:
    tf_rep = onnx_tf.backend.prepare(model)
    tf_rep.export_graph(_CFG['output_filename'])
    _del_location(_CFG['output_filename'])
    return ''
  except Exception as ex:
    _del_location(_CFG['output_filename'])
    strack_trace = str(ex).strip().split('\n')
    if len(strack_trace) > 1:
      err_msg = strack_trace[-1].strip()
      # OpUnsupportedException gets raised as a RuntimeError
      if 'OP_UNSUPPORTED_EXCEPT' in str(ex):
        err_msg = err_msg.replace(type(ex).__name__, 'OpUnsupportedException')
      return err_msg
    return '{}: {}'.format(type(ex).__name__, strack_trace[0].strip())


def _report_model(file_path, results=Results(), onnx_model_count=1):
  """Generate a report status for a single model, and append it to results."""
  size_pulled = _pull_model_file(file_path)
  if _CFG['dry_run']:
    ir_version = ''
    opset_version = ''
    check_err = ''
    convert_err = ''
    emoji_validated = ''
    emoji_converted = ''
    emoji_overall = ':heavy_minus_sign:'
    results.skip_count += 1
  else:
    if _CFG['verbose']:
      print('Testing', file_path)
    model = onnx.load(os.path.join(_CFG['models_dir'], file_path))
    ir_version = model.ir_version
    opset_version = model.opset_import[0].version
    check_err = _report_check_model(model)
    convert_err = '' if check_err else _report_convert_model(model)

    if not check_err and not convert_err:
      # https://github-emoji-list.herokuapp.com/
      # validation and conversion passed
      emoji_validated = ':ok:'
      emoji_converted = ':ok:'
      emoji_overall = ':heavy_check_mark:'
      results.pass_count += 1
    elif not check_err:
      # validation pass, but conversion did not
      emoji_validated = ':ok:'
      emoji_converted = convert_err
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
      emoji_overall = ':x:'
      results.fail_count += 1

  results.append_detail('{} | {}. {} | {} | {} | {} | {} | {}'.format(
      emoji_overall, onnx_model_count, file_path[file_path.rindex('/') + 1:],
      _size_with_units(size_pulled[0]), ir_version, opset_version,
      emoji_validated, emoji_converted))

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
  _CFG['output_filename'] = os.path.join(norm_output_dir, 'tmp_model.pb')
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
  _del_location(_CFG['output_filename'])

  # run tests first, but append to report after summary
  results = Results()
  for root, subdir, files in os.walk(_CFG['models_dir']):
    subdir.sort()
    if 'model' in subdir:
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
            'ONNX-TF Converted')
        results.append_detail(
            '------ | ----- | ---- | -- | ----- | ------------ | '
            '---------')
    onnx_model_count = 0
    for item in sorted(files):
      if item.endswith('.onnx'):
        file_path = os.path.relpath(os.path.join(root, item),
                                    _CFG['models_dir'])
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
