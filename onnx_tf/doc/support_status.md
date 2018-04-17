ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|ATen|N/A|
|Abs|1|
|Add|1|
|Affine|N/A|
|And|1|
|ArgMax|1|
|ArgMin|1|
|AveragePool|1|
|BatchNormalization|1, 6|
|Cast|1|
|Ceil|1|
|Clip|1|
|Concat|1, 4|
|Constant|1|
|ConstantFill|N/A|
|Conv|1|
|ConvTranspose|1|
|Crop|N/A|
|DepthToSpace|1|
|Div|1|
|Dropout|1|
|Elu|1|
|Equal|1|
|Exp|1|
|FC|N/A|
|Flatten|1|
|Floor|1|
|GRU|N/A|
|GRUUnit|N/A|
|Gather|1|
|Gemm|1|
|GivenTensorFill|N/A|
|GlobalAveragePool|1|
|GlobalLpPool|1, 2|
|GlobalMaxPool|1|
|Greater|1|
|HardSigmoid|1|
|Hardmax|1|
|Identity|1|
|If|N/A|
|ImageScaler|N/A|
|InstanceNormalization|N/A|
|LRN|1|
|LSTM|1|
|LeakyRelu|1|
|Less|1|
|Log|1|
|LogSoftmax|1|
|Loop|N/A|
|LoopIndexTensor|N/A|
|LpNormalization|1|
|LpPool|N/A|
|MatMul|1|
|Max|1|
|MaxPool|1|
|MaxRoiPool|N/A|
|Mean|1|
|MeanVarianceNormalization|N/A|
|Min|1|
|Mul|1|
|Neg|1|
|Not|1|
|Or|1|
|PRelu|1|
|Pad|1, 2|
|ParametricSoftplus|N/A|
|Pow|1|
|RNN|N/A|
|RandomNormal|1|
|RandomNormalLike|1|
|RandomUniform|1|
|RandomUniformLike|1|
|Reciprocal|1|
|ReduceL1|1|
|ReduceL2|N/A|
|ReduceLogSum|N/A|
|ReduceLogSumExp|1|
|ReduceMax|1|
|ReduceMean|1|
|ReduceMin|1|
|ReduceProd|1|
|ReduceSum|1|
|ReduceSumSquare|1|
|Relu|1|
|Reshape|1, 5|
|Scale|N/A|
|ScaledTanh|N/A|
|Selu|1|
|Shape|1|
|Sigmoid|1|
|Size|1|
|Slice|1|
|Softmax|1|
|Softplus|1|
|Softsign|1|
|SpaceToDepth|1|
|Split|1, 2|
|Sqrt|1|
|Squeeze|1|
|Sub|1|
|Sum|1|
|Tanh|1|
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
|add|1|
|avg_pool|1|
|bias_add|1|
|cast|1|
|concat_v2|1, 4|
|conv1_d|1|
|conv2_d|1|
|conv3_d|1|
|equal|1|
|fused_batch_norm|1, 6|
|greater|1|
|identity|1|
|less|1|
|logical_and|1|
|logical_not|1|
|logical_or|1|
|logical_xor|1|
|mat_mul|1|
|max|1|
|max_pool|1|
|mean|1|
|min|1|
|mul|1|
|pad|1, 2|
|pow|1|
|prod|1|
|random_standard_normal|1|
|random_uniform|1|
|real_div|1|
|reciprocal|1|
|relu|1|
|reshape|1, 5|
|shape|1|
|sigmoid|1|
|softmax|1|
|split_v|1, 2|
|sqrt|1|
|squeeze|1|
|sub|1|
|sum|1|
|tanh|1|
|transpose|1|
