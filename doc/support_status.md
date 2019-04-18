ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|ATen|N/A|
|Abs|1, 6|
|Acos|7|
|Acosh|9|
|Add|1, 6, 7|
|Affine|N/A|
|And|1, 7|
|ArgMax|1|
|ArgMin|1|
|Asin|7|
|Asinh|9|
|Atan|7|
|Atanh|9|
|AveragePool|1, 7|
|BatchNormalization|1, 6, 7, 9|
|Cast|1, 6, 9|
|Ceil|1, 6|
|Clip|1, 6|
|Compress|9|
|Concat|1, 4|
|Constant|1, 9|
|ConstantFill|1|
|ConstantOfShape|9|
|Conv|1|
|ConvTranspose|1|
|Cos|7|
|Cosh|9|
|Crop|N/A|
|DepthToSpace|1|
|Div|1, 6, 7|
|Dropout|1, 6, 7|
|DynamicSlice|1|
|Elu|1, 6|
|Equal|1, 7|
|Erf|9|
|Exp|1, 6|
|Expand|8|
|EyeLike|9|
|Flatten|1, 9|
|Floor|1, 6|
|GRU|1, 3, 7|
|GRUUnit|N/A|
|Gather|1|
|Gemm|1, 6, 7, 9|
|GivenTensorFill|N/A|
|GlobalAveragePool|1|
|GlobalLpPool|1, 2|
|GlobalMaxPool|1|
|Greater|1, 7, 9|
|HardSigmoid|1, 6|
|Hardmax|1|
|Identity|1|
|If|N/A|
|ImageScaler|1|
|InstanceNormalization|1, 6|
|IsNaN|9|
|LRN|1|
|LSTM|1, 7|
|LeakyRelu|1, 6|
|Less|1, 7, 9|
|Log|1, 6|
|LogSoftmax|1|
|Loop|N/A|
|LpNormalization|1|
|LpPool|N/A|
|MatMul|1, 9|
|Max|1, 6, 8|
|MaxPool|1, 8|
|MaxRoiPool|N/A|
|MaxUnpool|N/A|
|Mean|1, 6, 8|
|MeanVarianceNormalization|1|
|Min|1, 6, 8|
|Mul|1, 6, 7|
|Multinomial|N/A|
|Neg|1, 6|
|NonZero|9|
|Not|1|
|OneHot|N/A|
|Or|1, 7|
|PRelu|1, 6, 7, 9|
|Pad|1, 2|
|ParametricSoftplus|N/A|
|Pow|1, 7|
|RNN|1, 7|
|RandomNormal|1|
|RandomNormalLike|1|
|RandomUniform|1|
|RandomUniformLike|1|
|Reciprocal|1, 6|
|ReduceL1|1|
|ReduceL2|1|
|ReduceLogSum|1|
|ReduceLogSumExp|1|
|ReduceMax|1|
|ReduceMean|1|
|ReduceMin|1|
|ReduceProd|1|
|ReduceSum|1|
|ReduceSumSquare|1|
|Relu|1, 6|
|Reshape|1, 5|
|Scale|N/A|
|ScaledTanh|N/A|
|Scan|N/A|
|Scatter|N/A|
|Selu|1, 6|
|Shape|1|
|Shrink|9|
|Sigmoid|1, 6|
|Sign|9|
|Sin|7|
|Sinh|9|
|Size|1|
|Slice|1|
|Softmax|1|
|Softplus|1|
|Softsign|1|
|SpaceToDepth|1|
|Split|1, 2|
|Sqrt|1, 6|
|Squeeze|1|
|Sub|1, 6, 7|
|Sum|1, 6, 8|
|Tan|7|
|Tanh|1, 6|
|TfIdfVectorizer|N/A|
|ThresholdedRelu|1|
|Tile|1, 6|
|TopK|1|
|Transpose|1|
|Unsqueeze|1|
|Upsample|7, 9|
|Where|9|
|Xor|1, 7|


Frontend
______

| Tensorflow Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|Abs|1, 6|
|Acos|7|
|Acosh|9|
|Add|1, 6, 7|
|AddN|1, 6, 8|
|ArgMax|1|
|ArgMin|1|
|Asin|7|
|Asinh|9|
|Atan|7|
|Atanh|9|
|AvgPool|1, 7|
|BatchNorm|1, 6, 7, 9|
|BiasAdd|1, 6, 7|
|Cast|1, 6, 9|
|Ceil|1, 6|
|ConcatV2|1, 4|
|Conv1D|1|
|Conv2D|1|
|Conv3D|1|
|Cos|7|
|Cosh|9|
|DepthwiseConv2dNative|1|
|Equal|1, 7|
|Erf|9|
|Exp|1, 6|
|ExpandDims|1|
|Fill|1|
|Floor|1, 6|
|FloorDiv|1|
|FusedBatchNorm|1, 6, 7, 9|
|GRU|1, 3, 7|
|GatherV2|1|
|Greater|1, 7, 9|
|Identity|1|
|IsNan|9|
|LSTM|1, 7|
|Less|1, 7, 9|
|Log|1, 6|
|LogSoftmax|1|
|LogicalAnd|1, 7|
|LogicalNot|1|
|LogicalOr|1, 7|
|LogicalXor|1, 7|
|MatMul|1, 9|
|Max|1|
|MaxPool|1, 8|
|MaxPoolWithArgmax|8|
|Maximum|1, 6|
|Mean|1|
|Min|1|
|Minimum|1, 6|
|Mul|1, 6, 7|
|Neg|1, 6|
|Pack|1|
|Pad|1, 2|
|Pow|1, 7|
|Prod|1|
|RNN|1, 7|
|RandomStandardNormal|1|
|RandomUniform|1|
|RealDiv|1, 6, 7|
|Reciprocal|1, 6|
|Relu|1, 6|
|Relu6|1, 6|
|Reshape|1, 5|
|ResizeBilinear|9|
|Rsqrt|1|
|Select|9|
|Selu|1, 6|
|Shape|1|
|Sigmoid|1, 6|
|Sign|9|
|Sin|7|
|Sinh|9|
|Size|1|
|Slice|1|
|Softmax|1|
|Softplus|1|
|Softsign|1|
|SpaceToDepth|1|
|Split|1, 2|
|SplitV|1, 2|
|Sqrt|1, 6|
|Square|1|
|Squeeze|1|
|StridedSlice|1|
|Sub|1, 6, 7|
|Sum|1|
|Tan|7|
|Tanh|1, 6|
|Tile|6|
|TopKV2|1|
|Transpose|1|
|Unpack|1|
|ZerosLike|9|
