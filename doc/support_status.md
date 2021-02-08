# ONNX-Tensorflow Support Status
|||
|-:|:-|
|ONNX-Tensorflow Version|Master ( commit id: b8a88ded3dc16f438b9d6360a482d25ba8d620f7 )|
|ONNX Version|Master ( commit id: e1cf52650d29046bfe79e71f97ae8ee6b52bdd6f )|
|Tensorflow Version|v2.4.1|

Notes:
* Values that are new or updated from a previous opset version are in bold.
* -: not defined in corresponding ONNX opset version
* \*: the operator is deprecated
* :small_red_triangle:: not supported yet
* :small_orange_diamond:: partially supported
* the rest are all supported

|||||||||||||||||
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**ONNX Operator**|**Opset 1**|**Opset 2**|**Opset 3**|**Opset 4**|**Opset 5**|**Opset 6**|**Opset 7**|**Opset 8**|**Opset 9**|**Opset 10**|**Opset 11**|**Opset 12**|**Opset 13**|**Opset 14**|**ONNX Operator**|
|Abs|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Abs|
|Acos|-|-|-|-|-|-|**7**|7|7|7|7|7|7|7|Acos|
|Acosh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Acosh|
|Add|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**|13|Add|
|And|**1**|1|1|1|1|1|**7**|7|7|7|7|7|7|7|And|
|ArgMax|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**|**13**|13|ArgMax|
|ArgMin|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**|**13**|13|ArgMin|
|Asin|-|-|-|-|-|-|**7**|7|7|7|7|7|7|7|Asin|
|Asinh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Asinh|
|Atan|-|-|-|-|-|-|**7**|7|7|7|7|7|7|7|Atan|
|Atanh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Atanh|
|AveragePool|**1**|1|1|1|1|1|**7**|7|7|**10**|**11**|11|11|11|AveragePool|
|BatchNormalization|**1**|1|1|1|1|**6**|**7**|7|**9**|9|9|9|9|9|BatchNormalization|
|BitShift|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|BitShift|
|Cast|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**6**:small_orange_diamond:|6:small_orange_diamond:|6:small_orange_diamond:|**9**:small_orange_diamond:|9:small_orange_diamond:|9:small_orange_diamond:|9:small_orange_diamond:|**13**:small_orange_diamond:|13:small_orange_diamond:|Cast|
|Ceil|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Ceil|
|Celu|-|-|-|-|-|-|-|-|-|-|-|**12**|12|12|Celu|
|Clip|**1**|1|1|1|1|**6**|6|6|6|6|**11**|**12**|**13**|13|Clip|
|Compress|-|-|-|-|-|-|-|-|**9**|9|**11**|11|11|11|Compress|
|Concat|**1**|1|1|**4**|4|4|4|4|4|4|**11**|11|**13**|13|Concat|
|ConcatFromSequence|-|-|-|-|-|-|-|-|-|-|**11**:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|ConcatFromSequence|
|Constant|**1**|1|1|1|1|1|1|1|**9**|9|**11**|**12**|**13**|13|Constant|
|ConstantOfShape|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|ConstantOfShape|
|Conv|**1**|1|1|1|1|1|1|1|1|1|**11**|11|11|11|Conv|
|ConvInteger|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|ConvInteger|
|ConvTranspose|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**11**:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|ConvTranspose|
|Cos|-|-|-|-|-|-|**7**|7|7|7|7|7|7|7|Cos|
|Cosh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Cosh|
|CumSum|-|-|-|-|-|-|-|-|-|-|**11**|11|11|**14**:small_red_triangle:|CumSum|
|DepthToSpace|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|DepthToSpace|
|DequantizeLinear|-|-|-|-|-|-|-|-|-|**10**|10|10|**13**|13|DequantizeLinear|
|Det|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|Det|
|Div|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**|13|Div|
|Dropout|**1**|1|1|1|1|**6**|**7**|7|7|**10**|10|**12**|**13**|13|Dropout|
|DynamicQuantizeLinear|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|DynamicQuantizeLinear|
|Einsum|-|-|-|-|-|-|-|-|-|-|-|**12**|12|12|Einsum|
|Elu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|6|Elu|
|Equal|**1**|1|1|1|1|1|**7**|7|7|7|**11**|11|**13**|13|Equal|
|Erf|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**|13|Erf|
|Exp|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Exp|
|Expand|-|-|-|-|-|-|-|**8**|8|8|8|8|**13**|13|Expand|
|EyeLike|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|EyeLike|
|Flatten|**1**|1|1|1|1|1|1|1|**9**|9|**11**|11|**13**|13|Flatten|
|Floor|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Floor|
|GRU|**1**:small_orange_diamond:|1:small_orange_diamond:|**3**:small_orange_diamond:|3:small_orange_diamond:|3:small_orange_diamond:|3:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|GRU|
|Gather|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|Gather|
|GatherElements|-|-|-|-|-|-|-|-|-|-|**11**|11|**13**|13|GatherElements|
|GatherND|-|-|-|-|-|-|-|-|-|-|**11**|**12**|**13**|13|GatherND|
|Gemm|**1**|1|1|1|1|**6**|**7**|7|**9**|9|**11**|11|**13**|13|Gemm|
|GlobalAveragePool|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|GlobalAveragePool|
|GlobalLpPool|**1**|**2**|2|2|2|2|2|2|2|2|2|2|2|2|GlobalLpPool|
|GlobalMaxPool|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|GlobalMaxPool|
|Greater|**1**|1|1|1|1|1|**7**|7|**9**|9|9|9|**13**|13|Greater|
|GreaterOrEqual|-|-|-|-|-|-|-|-|-|-|-|**12**|12|12|GreaterOrEqual|
|HardSigmoid|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|6|HardSigmoid|
|Hardmax|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|Hardmax|
|Identity|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**13**:small_orange_diamond:|13:small_orange_diamond:|Identity|
|If|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|If|
|InstanceNormalization|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|6|InstanceNormalization|
|IsInf|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|IsInf|
|IsNaN|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**|13|IsNaN|
|LRN|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**|13|LRN|
|LSTM|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|LSTM|
|LeakyRelu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|6|LeakyRelu|
|Less|**1**|1|1|1|1|1|**7**|7|**9**|9|9|9|**13**|13|Less|
|LessOrEqual|-|-|-|-|-|-|-|-|-|-|-|**12**|12|12|LessOrEqual|
|Log|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Log|
|LogSoftmax|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|LogSoftmax|
|Loop|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|Loop|
|LpNormalization|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|LpNormalization|
|LpPool|**1**|**2**|2|2|2|2|2|2|2|2|**11**|11|11|11|LpPool|
|MatMul|**1**|1|1|1|1|1|1|1|**9**|9|9|9|**13**|13|MatMul|
|MatMulInteger|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|MatMulInteger|
|Max|**1**|1|1|1|1|**6**|6|**8**|8|8|8|**12**|**13**|13|Max|
|MaxPool|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**8**:small_orange_diamond:|8:small_orange_diamond:|**10**:small_orange_diamond:|**11**:small_orange_diamond:|**12**:small_orange_diamond:|12:small_orange_diamond:|12:small_orange_diamond:|MaxPool|
|MaxRoiPool|**1**:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|MaxRoiPool|
|MaxUnpool|-|-|-|-|-|-|-|-|**9**|9|**11**|11|11|11|MaxUnpool|
|Mean|**1**|1|1|1|1|**6**|6|**8**|8|8|8|8|**13**|13|Mean|
|MeanVarianceNormalization|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**|13|MeanVarianceNormalization|
|Min|**1**|1|1|1|1|**6**|6|**8**|8|8|8|**12**|**13**|13|Min|
|Mod|-|-|-|-|-|-|-|-|-|**10**|10|10|**13**|13|Mod|
|Mul|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**|13|Mul|
|Multinomial|-|-|-|-|-|-|**7**:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|7:small_red_triangle:|Multinomial|
|Neg|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Neg|
|NegativeLogLikelihoodLoss|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|**13**:small_red_triangle:|13:small_red_triangle:|NegativeLogLikelihoodLoss|
|NonMaxSuppression|-|-|-|-|-|-|-|-|-|**10**|**11**|11|11|11|NonMaxSuppression|
|NonZero|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**|13|NonZero|
|Not|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|Not|
|OneHot|-|-|-|-|-|-|-|-|**9**|9|**11**|11|11|11|OneHot|
|Or|**1**|1|1|1|1|1|**7**|7|7|7|7|7|7|7|Or|
|PRelu|**1**|1|1|1|1|**6**|**7**|7|**9**|9|9|9|9|9|PRelu|
|Pad|**1**|**2**|2|2|2|2|2|2|2|2|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|Pad|
|Pow|**1**|1|1|1|1|1|**7**|7|7|7|7|**12**|**13**|13|Pow|
|QLinearConv|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|QLinearConv|
|QLinearMatMul|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|QLinearMatMul|
|QuantizeLinear|-|-|-|-|-|-|-|-|-|**10**|10|10|**13**|13|QuantizeLinear|
|RNN|**1**:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|1:small_orange_diamond:|**7**:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|7:small_orange_diamond:|RNN|
|RandomNormal|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|RandomNormal|
|RandomNormalLike|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|RandomNormalLike|
|RandomUniform|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|RandomUniform|
|RandomUniformLike|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|RandomUniformLike|
|Range|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|Range|
|Reciprocal|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Reciprocal|
|ReduceL1|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|ReduceL1|
|ReduceL2|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|ReduceL2|
|ReduceLogSum|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|ReduceLogSum|
|ReduceLogSumExp|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|ReduceLogSumExp|
|ReduceMax|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**|**13**|13|ReduceMax|
|ReduceMean|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|ReduceMean|
|ReduceMin|**1**|1|1|1|1|1|1|1|1|1|**11**|**12**|**13**|13|ReduceMin|
|ReduceProd|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|ReduceProd|
|ReduceSum|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|ReduceSum|
|ReduceSumSquare|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|ReduceSumSquare|
|Relu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|**14**:small_red_triangle:|Relu|
|Reshape|**1**|1|1|1|**5**|5|5|5|5|5|5|5|**13**:small_red_triangle:|**14**:small_red_triangle:|Reshape|
|Resize|-|-|-|-|-|-|-|-|-|**10**:small_orange_diamond:|**11**:small_orange_diamond:|11:small_orange_diamond:|**13**:small_orange_diamond:|13:small_orange_diamond:|Resize|
|ReverseSequence|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|ReverseSequence|
|RoiAlign|-|-|-|-|-|-|-|-|-|**10**:small_orange_diamond:|10:small_orange_diamond:|10:small_orange_diamond:|10:small_orange_diamond:|10:small_orange_diamond:|RoiAlign|
|Round|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|Round|
|Scan|-|-|-|-|-|-|-|**8**|**9**|9|**11**|11|11|11|Scan|
|Scatter|-|-|-|-|-|-|-|-|**9**|9|**11**\*|11\*|11\*|11\*|Scatter|
|ScatterElements|-|-|-|-|-|-|-|-|-|-|**11**|11|**13**|13|ScatterElements|
|ScatterND|-|-|-|-|-|-|-|-|-|-|**11**|11|**13**|13|ScatterND|
|Selu|**1**|1|1|1|1|**6**|6|6|6|6|6|6|6|6|Selu|
|SequenceAt|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|SequenceAt|
|SequenceConstruct|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|SequenceConstruct|
|SequenceEmpty|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|SequenceEmpty|
|SequenceErase|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|SequenceErase|
|SequenceInsert|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|SequenceInsert|
|SequenceLength|-|-|-|-|-|-|-|-|-|-|**11**|11|11|11|SequenceLength|
|Shape|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**|13|Shape|
|Shrink|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Shrink|
|Sigmoid|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Sigmoid|
|Sign|-|-|-|-|-|-|-|-|**9**|9|9|9|**13**|13|Sign|
|Sin|-|-|-|-|-|-|**7**|7|7|7|7|7|7|7|Sin|
|Sinh|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Sinh|
|Size|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**|13|Size|
|Slice|**1**|1|1|1|1|1|1|1|1|**10**|**11**|11|**13**|13|Slice|
|Softmax|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**|13|Softmax|
|SoftmaxCrossEntropyLoss|-|-|-|-|-|-|-|-|-|-|-|**12**:small_red_triangle:|**13**:small_red_triangle:|13:small_red_triangle:|SoftmaxCrossEntropyLoss|
|Softplus|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|Softplus|
|Softsign|**1**|1|1|1|1|1|1|1|1|1|1|1|1|1|Softsign|
|SpaceToDepth|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**|13|SpaceToDepth|
|Split|**1**|**2**|2|2|2|2|2|2|2|2|**11**|11|**13**|13|Split|
|SplitToSequence|-|-|-|-|-|-|-|-|-|-|**11**:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|11:small_orange_diamond:|SplitToSequence|
|Sqrt|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Sqrt|
|Squeeze|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|Squeeze|
|StringNormalizer|-|-|-|-|-|-|-|-|-|**10**:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|10:small_red_triangle:|StringNormalizer|
|Sub|**1**|1|1|1|1|**6**|**7**|7|7|7|7|7|**13**|13|Sub|
|Sum|**1**|1|1|1|1|**6**|6|**8**|8|8|8|8|**13**|13|Sum|
|Tan|-|-|-|-|-|-|**7**|7|7|7|7|7|7|7|Tan|
|Tanh|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Tanh|
|TfIdfVectorizer|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|TfIdfVectorizer|
|ThresholdedRelu|-|-|-|-|-|-|-|-|-|**10**|10|10|10|10|ThresholdedRelu|
|Tile|**1**|1|1|1|1|**6**|6|6|6|6|6|6|**13**|13|Tile|
|TopK|**1**|1|1|1|1|1|1|1|1|**10**|**11**|11|11|11|TopK|
|Transpose|**1**|1|1|1|1|1|1|1|1|1|1|1|**13**|13|Transpose|
|Unique|-|-|-|-|-|-|-|-|-|-|**11**:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|11:small_red_triangle:|Unique|
|Unsqueeze|**1**|1|1|1|1|1|1|1|1|1|**11**|11|**13**:small_red_triangle:|13:small_red_triangle:|Unsqueeze|
|Upsample|**1**:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|1:small_red_triangle:|**7**:small_orange_diamond:|7:small_orange_diamond:|**9**:small_orange_diamond:|**10**\*|10\*|10\*|10\*|10\*|Upsample|
|Where|-|-|-|-|-|-|-|-|**9**|9|9|9|9|9|Where|
|Xor|**1**|1|1|1|1|1|**7**|7|7|7|7|7|7|7|Xor|

ONNX-TF Supported Operators / ONNX Operators: 146 / 162

Notes:
1. Cast: Cast string to data types other than float32/float64/int32/int64 is not supported in Tensorflow
2. ConcatFromSequence: new_axis=1 not supported in Tensorflow.
3. ConvTranspose: ConvTranspose with dilations != 1, or transposed convolution for 4D or higher are not supported in Tensorflow.
4. GRU: GRU with clip or GRU with linear_before_reset, or GRU not using sigmoid for z and r, or GRU using Elu as the activation function with alpha != 1, or GRU using HardSigmoid as the activation function with alpha != 0.2 or beta != 0.5 are not supported in TensorFlow.
5. Identity: Identity with sequence type is not supported in Tensorflow
6. LSTM: LSTM not using sigmoid for `f`, or LSTM not using the same activation for `g` and `h` are not supported in Tensorflow.
7. MaxPool: MaxPoolWithArgmax with pad is None or incompatible mode, or MaxPoolWithArgmax with 4D or higher input, or MaxPoolWithArgmax with column major are not supported in Tensorflow.
8. RNN: RNN with clip is not supported in Tensorflow.
9. Resize: Resize required 4D input in Tensorflow. For opset 11, only the following attributes and inputs conbination are supported in Tensorflow:
	1. mode=nearest, coordinate_transformation_mode=align_corners, nearest_mode=round_prefer_ceil, can use scales(*) or sizes.
	2. mode=nearest, coordinate_transformation_mode=asymmetric, nearest_mode=floor, can use scales(*) or sizes.
	3. mode=nearest, coordinate_transformation_mode=tf_half_pixel_for_nn, nearest_mode=floor, can use scales(*) or sizes.
	4. mode=linear, coordinate_transformation_mode=align_corners, can use scales(*) or sizes.
	5. mode=linear, coordinate_transformation_mode=asymmetric, can use scales(*) or sizes.
	6. mode=linear, coordinate_transformation_mode=half_pixel, can use scales(*) or sizes.
	7. mode=cubic, coordinate_transformation_mode=align_corners, cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) or sizes.
	8. mode=cubic, coordinate_transformation_mode=asymmetric, cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) or sizes.
	9. mode=cubic, coordinate_transformation_mode=half_pixel, cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) or sizes.
	10. mode=nearest, coordinate_transformation_mode=tf_crop_and_resize, extrapolation_value=any_float_value, nearest_mode=round_prefer_ceil, can use scales or sizes.
	11. mode=linear, coordinate_transformation_mode=tf_crop_and_resize, extrapolation_value=any_float_value, can use scales or sizes.
	- Note (*): The accuracy of your model will go down, if the height and the width of the new sizes(scales * origial sizes) are not in whole numbers.
10. RoiAlign: sampling_ratio <= 0 is not fully supported.
11. SplitToSequence: Scalar as the split input not supported.
12. Upsample: Upsample required 4D input in Tensorflow.
