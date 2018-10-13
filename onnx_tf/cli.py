import argparse
import sys

import onnx_tf.converter


def main():
  args = sys.argv[1:]
  parser = argparse.ArgumentParser(
      description="onnx-tensorflow command line tools")
  parser.add_argument("tool", choices=["convert"], help="Available tools.")
  if len(args) == 0:
    parser.parse_args(["-h"])
  cli_tool = parser.parse_args([args[0]])
  if cli_tool.tool == "convert":
    return onnx_tf.converter.main(args[1:])


if __name__ == '__main__':
  main()
