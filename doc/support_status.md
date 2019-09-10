# ONNX-Tensorflow Support Status
|||
|-:|:-|
|ONNX-Tensorflow Version|Master ( commit id: be1432c613fd118bc73f348a81329d98e9043d70 )|
|ONNX Version|Master ( commit id: 25cf73e5ff4beb0234d657c481223529acedd26f )|
|Tensorflow Version|v1.14.0|

Notes:
* Values that are new or updated from a previous opset version are in bold.
* -: not defined in corresponding ONNX opset version
* \*: the operator is deprecated
* :small_red_triangle:: not supported yet
* :small_orange_diamond:: partially supported
* the rest are all supported

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|Abs|1, 6|
|Acos|7|
|Acosh|9|
|Add|1, 6, 7|
|And|1, 7|
|ArgMax|1|
|ArgMin|1|
|ArrayFeatureExtractor|N/A|
|Asin|7|
|Asinh|9|
|Atan|7|
|Atanh|9|
|AveragePool|1, 7|
|BatchNormalization|1, 6, 7, 9|
|Binarizer|N/A|
|BitShift|N/A|
|Cast|1, 6, 9|
|CastMap|N/A|
|CategoryMapper|N/A|
|Ceil|1, 6|
|Clip|1, 6|
|Compress|9|
|Concat|1, 4|
|Constant|1, 9|
|ConstantFill|1|
|ConstantOfShape|9|
|Conv|1|
|ConvInteger|N/A|
|ConvTranspose|1|
|Cos|7|
|Cosh|9|
|CumSum|N/A|
|DepthToSpace|1, 11|
|DequantizeLinear|N/A|
|Det|N/A|
|DictVectorizer|N/A|
|Div|1, 6, 7|
|Dropout|1, 6, 7, 10|
|DynamicQuantizeLinear|N/A|
|Elu|1, 6|
|Equal|1, 7, 11|
|Erf|9|
|Exp|1, 6|
|Expand|8|
|EyeLike|9|
|FeatureVectorizer|N/A|
|Flatten|1, 9|
|Floor|1, 6|
|GRU|1, 3, 7|
|Gather|1|
|GatherElements|N/A|
|GatherND|N/A|
|Gemm|1, 6, 7, 9|
|GlobalAveragePool|1|
|GlobalLpPool|1, 2|
|GlobalMaxPool|1|
|Greater|1, 7, 9|
|HardSigmoid|1, 6|
|Hardmax|1|
|Identity|1|
|If|N/A|
|ImageScaler|1|
|Imputer|N/A|
|InstanceNormalization|1, 6|
|IsInf|10|
|IsNaN|9|
|LRN|1|
|LSTM|1, 7|
|LabelEncoder|N/A|
|LeakyRelu|1, 6|
|Less|1, 7, 9|
|LinearClassifier|N/A|
|LinearRegressor|N/A|
|Log|1, 6|
|LogSoftmax|1|
|Loop|N/A|
|LpNormalization|1|
|LpPool|N/A|
|MatMul|1, 9|
|MatMulInteger|N/A|
|Max|1, 6, 8|
|MaxPool|1, 8, 10|
|MaxRoiPool|N/A|
|MaxUnpool|9|
|Mean|1, 6, 8|
|MeanVarianceNormalization|1, 9|
|Min|1, 6, 8|
|Mod|10|
|Mul|1, 6, 7|
|Multinomial|N/A|
|Neg|1, 6|
|NonMaxSuppression|N/A|
|NonZero|9|
|Normalizer|N/A|
|Not|1|
|OneHot|9|
|OneHotEncoder|N/A|
|Or|1, 7|
|PRelu|1, 6, 7, 9|
|Pad|1, 2|
|Pow|1, 7|
|QLinearConv|N/A|
|QLinearMatMul|N/A|
|QuantizeLinear|N/A|
|RNN|1, 7|
|RandomNormal|1|
|RandomNormalLike|1|
|RandomUniform|1|
|RandomUniformLike|1|
|Range|N/A|
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
|Resize|10|
|ReverseSequence|N/A|
|RoiAlign|N/A|
|Round|N/A|
|SVMClassifier|N/A|
|SVMRegressor|N/A|
|Scaler|N/A|
|Scan|N/A|
|Scatter|N/A|
|ScatterElements|N/A|
|ScatterND|N/A|
|Selu|1, 6|
|Shape|1|
|Shrink|9|
|Sigmoid|1, 6|
|Sign|9|
|Sin|7|
|Sinh|9|
|Size|1|
|Slice|1, 10|
|Softmax|1|
|Softplus|1|
|Softsign|1|
|SpaceToDepth|1|
|Split|1, 2|
|Sqrt|1, 6|
|Squeeze|1|
|StringNormalizer|N/A|
|Sub|1, 6, 7|
|Sum|1, 6, 8|
|Tan|7|
|Tanh|1, 6|
|TfIdfVectorizer|N/A|
|ThresholdedRelu|1, 10|
|Tile|1, 6|
|TopK|1, 10, 11|
|Transpose|1|
|TreeEnsembleClassifier|N/A|
|TreeEnsembleRegressor|N/A|
|Unique|N/A|
|Unsqueeze|1|
|Upsample|7, 9|
|Where|9|
|Xor|1, 7|
|ZipMap|N/A|
