import argparse
import sys

import onnx_tf.converter
import onnx_tf.opr_checker

def main():
  args = sys.argv[1:]
  parser = argparse.ArgumentParser(
      description="ONNX-Tensorflow Command Line Interface")
  parser.add_argument(
      "command", choices=["convert", "check"], help="Available commands.")

  if len(args) == 0:
    parser.parse_args(["-h"])
  cli_tool = parser.parse_args([args[0]])
  if cli_tool.command == "convert":
    return onnx_tf.converter.main(args[1:])
  elif cli_tool.command == "check":
    return onnx_tf.opr_checker.main(args[1:])


if __name__ == '__main__':
  main()
