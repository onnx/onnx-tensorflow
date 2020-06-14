# ONNX-Tensorflow Support Status
|||
|-:|:-|
|ONNX-Tensorflow Version|Master ( commit id: e2917f8d2458b96332ace3ede45a3b8a32d2ec41 )|
|ONNX Version|Master ( commit id: 925b3657924c0c16cd20b54595f41e76159b03ab )|
|Tensorflow Version|v1.15.0|

Notes:
* Values that are new or updated from a previous opset version are in bold.
* -: not defined in corresponding ONNX opset version
* \*: the operator is deprecated
* :small_red_triangle:: not supported yet
* :small_orange_diamond:: partially supported
* the rest are all supported

|||||||||||||||
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**ONNX Operator**|**Opset 1**|**Opset 2**|**Opset 3**|**Opset 4**|**Opset 5**|**Opset 6**|**Opset 7**|**Opset 8**|**Opset 9**|**Opset 10**|**Opset 11**|**Opset 12**|**Opset 13**|
|Abs|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|Acos|-|-|-|-|-|-|**7**|7|7|7|7|7|7|
|Acosh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Add|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**:small_red_triangle:|
|And|**1**|1|1|1|1|1|**7**|7|7|7|7|7|7|
|ArgMax|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**|**13**:small_red_triangle:|
|ArgMin|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**|**13**:small_red_triangle:|
|Asin|-|-|-|-|-|-|**7**|7|7|7|7|7|7|
|Asinh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Atan|-|-|-|-|-|-|**7**|7|7|7|7|7|7|
|Atanh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|AveragePool|**1**|1|1|1|1|1|**7**|7|7|**10**|**11**|11|11|
|BatchNormalization|**1**|1|1|1|1|**6**|**7**|7|**9**|9|9|9|9|
|BitShift|-|-|-|-|-|-|-|-|-|-|**11**|11|11|
|Cast|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**6**:small_orange_diamond:|6:small_orange_diamond:|6:small_orange_diamond:|**9**:small_orange_diamond:|9:small_orange_diamond:|9:small_orange_diamond:|9:small_orange_diamond:|**13**:small_red_triangle:|
|Ceil|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|Celu|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|12:small_red_triangle:|
|Clip|**1**|1|1|1|1|**6**|6|6|6|6|**11**|**12**:small_red_triangle:|**13**:small_red_triangle:|
|Compress|-|-|-|-|-|-|-|-|**9**|9|**11**|11|11|
|Concat|**1**|1|1|**4**|4|4|4|4|4|4|**11**|11|**13**:small_red_triangle:|
|ConcatFromSequence|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|Constant|**1**|1|1|1|1|1|1|1|**9**|9|**11**|**12**|**13**:small_red_triangle:|
|ConstantOfShape|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Conv|**1**|1|1|1|1|1|1|1|1|1|**11**|11|11|
|ConvInteger|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|ConvTranspose|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**11**:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|
|Cos|-|-|-|-|-|-|**7**|7|7|7|7|7|7|
|Cosh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|CumSum|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|DepthToSpace|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|DequantizeLinear|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|Det|-|-|-|-|-|-|-|-|-|-|**11**|11|11|
|Div|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**:small_red_triangle:|
|Dropout|**1**|1|1|1|1|**6**|**7**|7|7|**10**|10|**12**:small_red_triangle:|**13**:small_red_triangle:|
|DynamicQuantizeLinear|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|Einsum|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|12:small_red_triangle:|
|Elu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|
|Equal|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|**11**:small_orange_diamond:|11:small_orange_diamond:|**13**:small_red_triangle:|
|Erf|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**:small_red_triangle:|
|Exp|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|Expand|-|-|-|-|-|-|-|**8**|8|8|8|8|**13**:small_red_triangle:|
|EyeLike|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Flatten|**1**|1|1|1|1|1|1|1|**9**|9|**11**|11|**13**:small_red_triangle:|
|Floor|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|GRU|**1**:small_orange_diamond:|1:small_orange_diamond:|**3**:small_orange_diamond:|3:small_orange_diamond:|3:small_orange_diamond:|3:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|
|Gather|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|GatherElements|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|**13**:small_red_triangle:|
|GatherND|-|-|-|-|-|-|-|-|-|-|**11**|**12**:small_red_triangle:|**13**:small_red_triangle:|
|Gemm|**1**|1|1|1|1|**6**|**7**|7|**9**|9|**11**|11|**13**:small_red_triangle:|
|GlobalAveragePool|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|GlobalLpPool|**1**|**2**|2|2|2|2|2|2|2|2|2|2|2|
|GlobalMaxPool|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|Greater|**1**|1|1|1|1|1|**7**|7|**9**|9|9|9|**13**:small_red_triangle:|
|GreaterOrEqual|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|12:small_red_triangle:|
|HardSigmoid|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|
|Hardmax|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|Identity|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**:small_red_triangle:|
|If|**1**:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|InstanceNormalization|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|
|IsInf|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|IsNaN|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**:small_red_triangle:|
|LRN|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**:small_red_triangle:|
|LSTM|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|
|LeakyRelu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|
|Less|**1**|1|1|1|1|1|**7**|7|**9**|9|9|9|**13**:small_red_triangle:|
|LessOrEqual|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|12:small_red_triangle:|
|Log|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|LogSoftmax|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|Loop|**1**:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|LpNormalization|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|LpPool|**1**:small_red_triangle:|**2**:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|2:small_red_triangle:|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|MatMul|**1**|1|1|1|1|1|1|1|**9**|9|9|9|**13**:small_red_triangle:|
|MatMulInteger|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|Max|**1**|1|1|1|1|**6**|6|**8**|8|8|8|**12**:small_red_triangle:|**13**:small_red_triangle:|
|MaxPool|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**8**:small_orange_diamond:|8:small_orange_diamond:|**10**:small_orange_diamond:|**11**:small_orange_diamond:|**12**:small_red_triangle:|12:small_red_triangle:|
|MaxRoiPool|**1**:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|
|MaxUnpool|-|-|-|-|-|-|-|-|**9**|9|**11**|11|11|
|Mean|**1**|1|1|1|1|**6**|6|**8**|8|8|8|8|**13**:small_red_triangle:|
|MeanVarianceNormalization|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**:small_red_triangle:|
|Min|**1**|1|1|1|1|**6**|6|**8**|8|8|8|**12**:small_red_triangle:|**13**:small_red_triangle:|
|Mod|-|-|-|-|-|-|-|-|-|**10**:small_orange_diamond:|10:small_orange_diamond:|10:small_orange_diamond:|**13**:small_red_triangle:|
|Mul|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**:small_red_triangle:|
|Multinomial|-|-|-|-|-|-|**7**:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|
|Neg|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|NegativeLogLikelihoodLoss|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|12:small_red_triangle:|
|NonMaxSuppression|-|-|-|-|-|-|-|-|-|**10**|**11**|11|11|
|NonZero|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**:small_red_triangle:|
|Not|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|OneHot|-|-|-|-|-|-|-|-|**9**:small_orange_diamond:|9:small_orange_diamond:|**11**:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|
|Or|**1**|1|1|1|1|1|**7**|7|7|7|7|7|7|
|PRelu|**1**|1|1|1|1|**6**|**7**|7|**9**|9|9|9|9|
|Pad|**1**|**2**|2|2|2|2|2|2|2|2|**11**|11|**13**:small_red_triangle:|
|Pow|**1**|1|1|1|1|1|**7**|7|7|7|7|**12**:small_red_triangle:|**13**:small_red_triangle:|
|QLinearConv|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|QLinearMatMul|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|QuantizeLinear|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|RNN|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|
|RandomNormal|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|RandomNormalLike|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|RandomUniform|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|RandomUniformLike|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|Range|-|-|-|-|-|-|-|-|-|-|**11**|11|11|
|Reciprocal|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|ReduceL1|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceL2|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceLogSum|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceLogSumExp|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceMax|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**:small_red_triangle:|**13**:small_red_triangle:|
|ReduceMean|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceMin|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**:small_red_triangle:|**13**:small_red_triangle:|
|ReduceProd|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceSum|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|ReduceSumSquare|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|Relu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|Reshape|**1**|1|1|1|**5**|5|5|5|5|5|5|5|**13**:small_red_triangle:|
|Resize|-|-|-|-|-|-|-|-|-|**10**:small_orange_diamond:|**11**:small_red_triangle:|11:small_red_triangle:|**13**:small_red_triangle:|
|ReverseSequence|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|RoiAlign|-|-|-|-|-|-|-|-|-|**10**:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|
|Round|-|-|-|-|-|-|-|-|-|-|**11**|11|11|
|Scan|-|-|-|-|-|-|-|**8**|**9**|9|**11**|11|11|
|Scatter|-|-|-|-|-|-|-|-|**9**|9|**11**:small_red_triangle:|11:small_red_triangle:|**13**\*|
|ScatterElements|-|-|-|-|-|-|-|-|-|-|**11**|11|**13**:small_red_triangle:|
|ScatterND|-|-|-|-|-|-|-|-|-|-|**11**|11|**13**:small_red_triangle:|
|Selu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|
|SequenceAt|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|SequenceConstruct|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|SequenceEmpty|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|SequenceErase|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|SequenceInsert|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|SequenceLength|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|Shape|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**:small_red_triangle:|
|Shrink|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Sigmoid|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|Sign|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**:small_red_triangle:|
|Sin|-|-|-|-|-|-|**7**|7|7|7|7|7|7|
|Sinh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Size|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**:small_red_triangle:|
|Slice|**1**|1|1|1|1|1|1|1|1|**10**|**11**|11|**13**|
|Softmax|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|SoftmaxCrossEntropyLoss|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|**13**:small_red_triangle:|
|Softplus|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|Softsign|**1**|1|1|1|1|1|1|1|1|1|1|1|1|
|SpaceToDepth|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**:small_red_triangle:|
|Split|**1**|**2**|2|2|2|2|2|2|2|2|**11**|11|**13**:small_red_triangle:|
|SplitToSequence|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|Sqrt|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|Squeeze|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|StringNormalizer|-|-|-|-|-|-|-|-|-|**10**:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|
|Sub|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**:small_red_triangle:|
|Sum|**1**|1|1|1|1|**6**|6|**8**|8|8|8|8|**13**:small_red_triangle:|
|Tan|-|-|-|-|-|-|**7**|7|7|7|7|7|7|
|Tanh|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|TfIdfVectorizer|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|ThresholdedRelu|-|-|-|-|-|-|-|-|-|**10**|10|10|10|
|Tile|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**:small_red_triangle:|
|TopK|**1**|1|1|1|1|1|1|1|1|**10**|**11**|11|11|
|Transpose|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**:small_red_triangle:|
|Unique|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|
|Unsqueeze|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|
|Upsample|**1**:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|**7**:small_orange_diamond:|7:small_orange_diamond:|**9**:small_orange_diamond:|**10**:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|**13**\*|
|Where|-|-|-|-|-|-|-|-|**9**|9|9|9|9|
|Xor|**1**|1|1|1|1|1|**7**|7|7|7|7|7|7|

ONNX-TF Supported Operators / ONNX Operators: 65 / 162

Notes:
1. Cast: Cast string to float32/float64/int32/int64 are not supported in Tensorflow.
2. ConvTranspose: ConvTranspose with dilations != 1, or transposed convolution for 4D or higher are not supported in Tensorflow.
3. Equal: Equal inputs in uint16/uint32/uint64 are not supported in Tensorflow.
4. GRU: GRU with clip or GRU with linear_before_reset, or GRU not using sigmoid for z and r, or GRU using Elu as the activation function with alpha != 1, or GRU using HardSigmoid as the activation function with alpha != 0.2 or beta != 0.5 are not supported in TensorFlow.
5. LSTM: LSTM not using sigmoid for `f`, or LSTM not using the same activation for `g` and `h` are not supported in Tensorflow.
6. MaxPool: MaxPoolWithArgmax with pad is None or incompatible mode, or MaxPoolWithArgmax with 4D or higher input, orMaxPoolWithArgmax with column major are not supported in Tensorflow.
7. Mod: Mod Dividend or Divisor in int8/int16/uint8/uint16/uint32/uint64 are not supported in Tensorflow.
8. OneHot: OneHot indices in uint16/uint32/uint64/int8/int16/float16/float/double, or OneHot depth in uint8/uint16/uint32/uint64/int8/int16/int64/float16/float/double are not supported in Tensorflow.
9. RNN: RNN with clip is not supported in Tensorflow.
10. Resize: Resize required 4D input in Tensorflow.
11. Upsample: Upsample required 4D input in Tensorflow.
