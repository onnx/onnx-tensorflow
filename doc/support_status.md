ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|ATen|N/A|
|Abs|1, 6|
|Acos|7|
|Add|1, 6, 7|
|Affine|N/A|
|And|1, 7|
|ArgMax|1|
|ArgMin|1|
|Asin|7|
|Atan|7|
|AveragePool|1, 7|
|BatchNormalization|1, 6, 7|
|Cast|1, 6|
|Ceil|1, 6|
|Clip|1, 6|
|Concat|1, 4|
|Constant|1|
|ConstantFill|1|
|Conv|1|
|ConvTranspose|1|
|Cos|7|
|Crop|N/A|
|DepthToSpace|1|
|Div|1, 6, 7|
|Dropout|1, 6, 7|
|Elu|1, 6|
|Equal|1, 7|
|Exp|1, 6|
|Expand|8|
|Flatten|1|
|Floor|1, 6|
|GRU|1, 3, 7|
|GRUUnit|N/A|
|Gather|1|
|Gemm|1, 6, 7|
|GivenTensorFill|N/A|
|GlobalAveragePool|1|
|GlobalLpPool|1, 2|
|GlobalMaxPool|1|
|Greater|1, 7|
|HardSigmoid|1, 6|
|Hardmax|1|
|Identity|1|
|If|N/A|
|ImageScaler|1|
|InstanceNormalization|1, 6|
|LRN|1|
|LSTM|1, 7|
|LeakyRelu|1, 6|
|Less|1, 7|
|Log|1, 6|
|LogSoftmax|1|
|Loop|N/A|
|LoopIndexTensor|N/A|
|LpNormalization|1|
|LpPool|N/A|
|MatMul|1|
|Max|1, 6, 8|
|MaxPool|1, 8|
|MaxRoiPool|N/A|
|Mean|1, 6, 8|
|MeanVarianceNormalization|N/A|
|Min|1, 6, 8|
|Mul|1, 6, 7|
|Multinomial|N/A|
|Neg|1, 6|
|Not|1|
|Or|1, 7|
|PRelu|1, 6, 7|
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
|Selu|1, 6|
|Shape|1|
|Sigmoid|1, 6|
|Sin|7|
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
|ThresholdedRelu|1|
|Tile|1, 6|
|TopK|1|
|Transpose|1|
|Unsqueeze|1|
|Upsample|7|
|Xor|1, 7|


Frontend
______

| Tensorflow Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|abs|1, 6|
|acos|7|
|add|1, 6, 7|
|add_n|1, 6|
|arg_max|1|
|arg_min|1|
|asin|7|
|atan|7|
|avg_pool|1, 7|
|batch_norm|1, 6, 7|
|bias_add|1, 6, 7|
|cast|1, 6|
|ceil|1, 6|
|concat_v2|1, 4|
|conv1_d|1|
|conv2_d|1|
|conv3_d|1|
|cos|7|
|equal|1, 7|
|exp|1, 6|
|expand_dims|1|
|fill|1|
|floor|1, 6|
|fused_batch_norm|1, 6, 7|
|greater|1, 7|
|identity|1|
|less|1, 7|
|log|1, 6|
|log_softmax|1|
|logical_and|1, 7|
|logical_not|1|
|logical_or|1, 7|
|logical_xor|1, 7|
|mat_mul|1|
|max|1|
|max_pool|1, 8|
|max_pool_with_argmax|8|
|maximum|1, 6|
|mean|1|
|min|1|
|minimum|1, 6|
|mul|1, 6, 7|
|neg|1, 6|
|pack|1|
|pad|1, 2|
|pow|1, 7|
|prod|1|
|random_standard_normal|1|
|random_uniform|1|
|real_div|1, 6, 7|
|reciprocal|1, 6|
|relu|1, 6|
|reshape|1, 5|
|selu|1, 6|
|shape|1|
|sigmoid|1, 6|
|sin|7|
|slice|1|
|softmax|1|
|softplus|1|
|softsign|1|
|space_to_depth|1|
|split_v|1, 2|
|sqrt|1, 6|
|squeeze|1|
|sub|1, 6, 7|
|sum|1|
|tan|7|
|tanh|1, 6|
|tile|6|
|top_k_v2|1|
|transpose|1|
|unpack|1|
