ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|ATen|N/A|
|Abs|1, 6|
|Acos|N/A|
|Add|1, 6|
|Affine|N/A|
|And|1|
|ArgMax|1|
|ArgMin|1|
|Asin|N/A|
|Atan|N/A|
|AveragePool|1, 7|
|BatchNormalization|1, 6|
|Cast|1|
|Ceil|1, 6|
|Clip|1, 6|
|Concat|1, 4|
|Constant|1|
|ConstantFill|1|
|Conv|1|
|ConvTranspose|1|
|Cos|N/A|
|Crop|N/A|
|DepthToSpace|1|
|Div|1, 6|
|Dropout|1, 6|
|Elu|1, 6|
|Equal|1|
|Exp|1, 6|
|Flatten|1|
|Floor|1, 6|
|GRU|N/A|
|GRUUnit|N/A|
|Gather|1|
|Gemm|1, 6|
|GivenTensorFill|N/A|
|GlobalAveragePool|1|
|GlobalLpPool|1, 2|
|GlobalMaxPool|1|
|Greater|1|
|HardSigmoid|1, 6|
|Hardmax|1|
|Identity|1|
|If|N/A|
|ImageScaler|N/A|
|InstanceNormalization|N/A|
|LRN|1|
|LSTM|1|
|LeakyRelu|1, 6|
|Less|1|
|Log|1, 6|
|LogSoftmax|1|
|Loop|N/A|
|LoopIndexTensor|N/A|
|LpNormalization|1|
|LpPool|N/A|
|MatMul|1|
|Max|1, 6|
|MaxPool|1|
|MaxRoiPool|N/A|
|Mean|1, 6|
|MeanVarianceNormalization|N/A|
|Min|1, 6|
|Mul|1, 6|
|Neg|1, 6|
|Not|1|
|Or|1|
|PRelu|1, 6|
|Pad|1, 2|
|ParametricSoftplus|N/A|
|Pow|1|
|RNN|N/A|
|RandomNormal|1|
|RandomNormalLike|1|
|RandomUniform|1|
|RandomUniformLike|1|
|Reciprocal|1, 6|
|ReduceL1|1|
|ReduceL2|N/A|
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
|Selu|1, 6|
|Shape|1|
|Sigmoid|1, 6|
|Sin|N/A|
|Size|1|
|Slice|1|
|Softmax|1|
|Softplus|1|
|Softsign|1|
|SpaceToDepth|1|
|Split|1, 2|
|Sqrt|1, 6|
|Squeeze|1|
|Sub|1, 6|
|Sum|1, 6|
|Tan|N/A|
|Tanh|1, 6|
|ThresholdedRelu|1|
|Tile|1|
|TopK|1|
|Transpose|1|
|Unsqueeze|1|
|Upsample|1|
|Xor|1|


Frontend
______

| Tensorflow Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|Abs|1, 6|
|Acos|7|
|Add|1, 6|
|AddN|1, 6|
|ArgMax|1|
|ArgMin|1|
|Asin|7|
|Atan|7|
|AvgPool|1, 7|
|BatchNorm|1, 6|
|BiasAdd|1, 6|
|Cast|1|
|Ceil|1, 6|
|ConcatV2|1, 4|
|Conv1D|1|
|Conv2D|1|
|Conv3D|1|
|Cos|7|
|Exp|1, 6|
|ExpandDims|1|
|Fill|1|
|Floor|1, 6|
|FusedBatchNorm|1, 6|
|Log|1, 6|
|LogSoftmax|1|
|LogicalAnd|1|
|LogicalNot|1|
|LogicalOr|1|
|LogicalXor|1|
|Max|1|
|MaxPool|1|
|Maximun|1, 6|
|Mean|1|
|Min|1|
|Minimum|1, 6|
|Mul|1, 6|
|Neg|1, 6|
|Pad|1, 2|
|Pow|1|
|Prod|1|
|RandomStandardNormal|1|
|RandomUniform|1|
|RealDiv|1, 6|
|Reciprocal|1, 6|
|Relu|1, 6|
|Reshape|1, 5|
|Selu|1, 6|
|Sigmoid|1, 6|
|Sin|7|
|Softmax|1|
|Softplus|1|
|Softsign|1|
|SplitV|1, 2|
|Sqrt|1, 6|
|Squeeze|1|
|Sub|1, 6|
|Sum|1|
|Tan|7|
|Tanh|1, 6|
|Tile|1|
|TopKV2|1|
|Transpose|1|
