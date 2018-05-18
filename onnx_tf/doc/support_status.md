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
|FC|N/A|
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
|MaxPool|1, 7|
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
|add|1, 6|
|arg_max|1|
|arg_min|1|
|avg_pool|1, 7|
|bias_add|1, 6|
|cast|1|
|ceil|1, 6|
|concat_v2|1, 4|
|conv1_d|1|
|conv2_d|1|
|conv3_d|1|
|equal|1|
|exp|1, 6|
|fill|1|
|floor|1, 6|
|fused_batch_norm|1, 6|
|greater|1|
|identity|1|
|less|1|
|log|1, 6|
|log_softmax|1|
|logical_and|1, 6|
|logical_not|1|
|logical_or|1|
|logical_xor|1|
|mat_mul|1|
|max|1|
|max_pool|1, 7|
|maximum|1|
|mean|1|
|min|1|
|mul|1, 6|
|pad|1, 2|
|pow|1|
|prod|1|
|random_standard_normal|1|
|random_uniform|1|
|real_div|1, 6|
|reciprocal|1, 6|
|relu|1, 6|
|reshape|1, 5|
|shape|1|
|sigmoid|1, 6|
|softmax|1|
|space_to_depth|1|
|split_v|1, 2|
|sqrt|1, 6|
|squeeze|1|
|sub|1, 6|
|sum|1|
|tanh|1, 6|
|tile|1|
|transpose|1|
