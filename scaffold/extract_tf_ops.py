from __future__ import print_function
import re
import pprint
import urllib2

def get_tf_defs():
	tf_op_defs = {}

	def clean(name, input):
		return input.replace(name + "(\"", "").replace("\")", "").split(":")[0].strip();

	url = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/ops/{}.cc"
	fnames = ["nn_ops", "math_ops", "array_ops"]

	for fname in fnames:

		content = urllib2.urlopen(url.format(fname)).read()
		content = re.sub("\([\s\n]+\"", "(\"", content)
		content = re.sub("\"[\s\n]+\"", "", content)
		content = content.split("\n")
		content = [x.strip() for x in content]

		in_op_def = False
		curr_op = {}
		for line in content:
			if (in_op_def):
				if line.startswith(".Input"):
					name = clean(".Input", line)
					# print("\ti " + name)
					curr_op["i"].append(name)
				elif line.startswith(".Output"):
					name = clean(".Output", line)
					# print("\to " + name)
					curr_op["o"].append(name)
				elif line.startswith(".Attr"):
					name = clean(".Attr", line)
					# print("\ta " + name)
					curr_op["a"].append(name)
				else:
					in_op_def = False
					tf_op_defs[curr_op["n"]] = curr_op
			else:
				if (line.startswith("REGISTER_OP")):
					in_op_def = True
					name = clean("REGISTER_OP", line)
					curr_op = {
						"n": name,
						"i": [],
						"o": [],
						"a": []
					}
					# print(name)
	return tf_op_defs

tf_op_defs = get_tf_defs()
pp = pprint.PrettyPrinter(indent=2)
# pp.pprint(tf_op_defs)

from onnx import onnx_pb2, helper, defs
node_def = helper.make_node("Relu", ["X"], ["Y"])
print(node_def)

all_schemas = defs.get_all_schemas()
# print(all_schemas)
# for name in all_schemas:
# 	# print(dir(all_schemas[name]))
# 	print("i", all_schemas[name].input_desc)
# 	print("o", all_schemas[name].output_desc)
# 	print("a", all_schemas[name].attributes)
all_schemas = dict(filter(lambda x: len(x[1].input_desc)==1 ,all_schemas.items()))
all_schemas = dict(filter(lambda x: len(x[1].attributes)==0 or len(x[1].attributes)==1,all_schemas.items()))

intersection = (set(tf_op_defs.keys()).intersection(set(all_schemas.keys())))

for key in intersection:
	print(key)
	pp.pprint(tf_op_defs[key])
	print("i", all_schemas[key].input_desc)
	print("o", all_schemas[key].output_desc)
	print("a", all_schemas[key].attributes)


# print(intersection)



