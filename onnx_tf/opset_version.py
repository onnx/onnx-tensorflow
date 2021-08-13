backend_opset_version = {
    'Abs': [1, 6, 13],
    'Acos': [7],
    'Acosh': [9],
    'Adagrad': [],
    'Adam': [],
    'Add': [1, 6, 7, 13, 14],
    'And': [1, 7],
    'ArgMax': [1, 11, 12, 13],
    'ArgMin': [1, 11, 12, 13],
    'ArrayFeatureExtractor': [],
    'Asin': [7],
    'Asinh': [9],
    'Atan': [7],
    'Atanh': [9],
    'AveragePool': [1, 7, 10, 11],
    'BatchNormalization': [1, 6, 7, 9, 14],
    'Bernoulli': [],
    'Binarizer': [],
    'BitShift': [11],
    'Cast': [1, 6, 9, 13],
    'CastLike': [],
    'CastMap': [],
    'CategoryMapper': [],
    'Ceil': [1, 6, 13],
    'Celu': [12],
    'Clip': [1, 6, 11, 12, 13],
    'Compress': [9, 11],
    'Concat': [1, 4, 11, 13],
    'ConcatFromSequence': [11],
    'Constant': [1, 9, 11, 12, 13],
    'ConstantFill': [1],
    'ConstantOfShape': [9],
    'Conv': [1, 11],
    'ConvInteger': [10],
    'ConvTranspose': [1, 11],
    'Cos': [7],
    'Cosh': [9],
    'CumSum': [11, 14],
    'DepthToSpace': [1, 11, 13],
    'DequantizeLinear': [10, 13],
    'Det': [11],
    'DictVectorizer': [],
    'Div': [1, 6, 7, 13, 14],
    'Dropout': [1, 6, 7, 10, 12, 13],
    'DynamicQuantizeLinear': [11],
    'Einsum': [12],
    'Elu': [1, 6],
    'Equal': [1, 7, 11, 13],
    'Erf': [9, 13],
    'Exp': [1, 6, 13],
    'Expand': [8, 13],
    'EyeLike': [9],
    'FeatureVectorizer': [],
    'Flatten': [1, 9, 11, 13],
    'Floor': [1, 6, 13],
    'GRU': [1, 3, 7, 14],
    'Gather': [1, 11, 13],
    'GatherElements': [11, 13],
    'GatherND': [11, 12, 13],
    'Gemm': [1, 6, 7, 9, 11, 13],
    'GlobalAveragePool': [1],
    'GlobalLpPool': [1, 2],
    'GlobalMaxPool': [1],
    'Gradient': [],
    'Greater': [1, 7, 9, 13],
    'GreaterOrEqual': [12],
    'HardSigmoid': [1, 6],
    'HardSwish': [14],
    'Hardmax': [1, 11, 13],
    'Identity': [1, 13, 14],
    'If': [1, 11, 13],
    'ImageScaler': [1],
    'Imputer': [],
    'InstanceNormalization': [1, 6],
    'IsInf': [10],
    'IsNaN': [9, 13],
    'LRN': [1, 13],
    'LSTM': [1, 7, 14],
    'LabelEncoder': [],
    'LeakyRelu': [1, 6],
    'Less': [1, 7, 9, 13],
    'LessOrEqual': [12],
    'LinearClassifier': [],
    'LinearRegressor': [],
    'Log': [1, 6, 13],
    'LogSoftmax': [1, 11, 13],
    'Loop': [1, 11, 13],
    'LpNormalization': [1],
    'LpPool': [1, 2, 11],
    'MatMul': [1, 9, 13],
    'MatMulInteger': [10],
    'Max': [1, 6, 8, 12, 13],
    'MaxPool': [1, 8, 10, 11, 12],
    'MaxRoiPool': [],
    'MaxUnpool': [9, 11],
    'Mean': [1, 6, 8, 13],
    'MeanVarianceNormalization': [1, 9, 13],
    'Min': [1, 6, 8, 12, 13],
    'Mod': [10, 13],
    'Momentum': [],
    'Mul': [1, 6, 7, 13, 14],
    'Multinomial': [7],
    'Neg': [1, 6, 13],
    'NegativeLogLikelihoodLoss': [],
    'NonMaxSuppression': [10, 11],
    'NonZero': [9, 13],
    'Normalizer': [],
    'Not': [1],
    'OneHot': [9, 11],
    'OneHotEncoder': [],
    'Optional': [],
    'OptionalGetElement': [],
    'OptionalHasElement': [],
    'Or': [1, 7],
    'PRelu': [1, 6, 7, 9],
    'Pad': [1, 2, 11, 13],
    'Pow': [1, 7, 12, 13],
    'QLinearConv': [10],
    'QLinearMatMul': [10],
    'QuantizeLinear': [10, 13],
    'RNN': [1, 7, 14],
    'RandomNormal': [1],
    'RandomNormalLike': [1],
    'RandomUniform': [1],
    'RandomUniformLike': [1],
    'Range': [11],
    'Reciprocal': [1, 6, 13],
    'ReduceL1': [1, 11, 13],
    'ReduceL2': [1, 11, 13],
    'ReduceLogSum': [1, 11, 13],
    'ReduceLogSumExp': [1, 11, 13],
    'ReduceMax': [1, 11, 12, 13],
    'ReduceMean': [1, 11, 13],
    'ReduceMin': [1, 11, 12, 13],
    'ReduceProd': [1, 11, 13],
    'ReduceSum': [1, 11, 13],
    'ReduceSumSquare': [1, 11, 13],
    'Relu': [1, 6, 13, 14],
    'Reshape': [1, 5, 13, 14],
    'Resize': [10, 11, 13],
    'ReverseSequence': [10],
    'RoiAlign': [10],
    'Round': [11],
    'SVMClassifier': [],
    'SVMRegressor': [],
    'Scaler': [],
    'Scan': [8, 9, 11],
    'Scatter': [9],
    'ScatterElements': [11, 13],
    'ScatterND': [11, 13],
    'Selu': [1, 6],
    'SequenceAt': [11],
    'SequenceConstruct': [11],
    'SequenceEmpty': [11],
    'SequenceErase': [11],
    'SequenceInsert': [11],
    'SequenceLength': [11],
    'Shape': [1, 13],
    'Shrink': [9],
    'Sigmoid': [1, 6, 13],
    'Sign': [9, 13],
    'Sin': [7],
    'Sinh': [9],
    'Size': [1, 13],
    'Slice': [1, 10, 11, 13],
    'Softmax': [1, 11, 13],
    'SoftmaxCrossEntropyLoss': [],
    'Softplus': [1],
    'Softsign': [1],
    'SpaceToDepth': [1, 13],
    'Split': [1, 2, 11, 13],
    'SplitToSequence': [11],
    'Sqrt': [1, 6, 13],
    'Squeeze': [1, 11],
    'StringNormalizer': [],
    'Sub': [1, 6, 7, 13, 14],
    'Sum': [1, 6, 8, 13],
    'Tan': [7],
    'Tanh': [1, 6, 13],
    'TfIdfVectorizer': [9],
    'ThresholdedRelu': [1, 10],
    'Tile': [1, 6, 13],
    'TopK': [1, 10, 11],
    'Transpose': [1, 13],
    'TreeEnsembleClassifier': [],
    'TreeEnsembleRegressor': [],
    'Trilu': [14],
    'Unique': [],
    'Unsqueeze': [1, 11],
    'Upsample': [7, 9],
    'Where': [9],
    'Xor': [1, 7],
    'ZipMap': []
}

backend_partial_support = {
    'BatchNormalization': 'BatchNormalization with training_mode=1 is not '
                          'supported in Tensorflow converte.',
    'Cast': 'Cast string to data types other than float32/float64/int32/int64 '
            'is not supported in Tensorflow',
    'ConcatFromSequence': 'new_axis=1 not supported in Tensorflow.',
    'ConvTranspose': 'ConvTranspose with dilations != 1, or transposed '
                     'convolution for 4D or higher are not supported in '
                     'Tensorflow.',
    'GRU': 'GRU with clip or GRU with linear_before_reset, or GRU not using '
           'sigmoid for z and r, or GRU using Elu as the activation function '
           'with alpha != 1, or GRU using HardSigmoid as the activation '
           'function with alpha != 0.2 or beta != 0.5 are not supported in '
           'TensorFlow.',
    'LSTM': 'LSTM not using sigmoid for `f`, or LSTM not using the same '
            'activation for `g` and `h` are not supported in Tensorflow.',
    'MaxPool': 'MaxPoolWithArgmax with pad is None or incompatible mode, or '
               'MaxPoolWithArgmax with 4D or higher input, or '
               'MaxPoolWithArgmax with column major are not supported in '
               'Tensorflow.',
    'RNN': 'RNN with clip is not supported in Tensorflow.',
    'Resize': 'Resize required 4D input in Tensorflow. For opset 11, only the '
              'following attributes and inputs conbination are supported in '
              'Tensorflow:\n'
              '\t1. mode=nearest, '
              'coordinate_transformation_mode=align_corners, '
              'nearest_mode=round_prefer_ceil, can use scales(*) or sizes.\n'
              '\t2. mode=nearest, coordinate_transformation_mode=asymmetric, '
              'nearest_mode=floor, can use scales(*) or sizes.\n'
              '\t3. mode=nearest, '
              'coordinate_transformation_mode=tf_half_pixel_for_nn, '
              'nearest_mode=floor, can use scales(*) or sizes.\n'
              '\t4. mode=linear, coordinate_transformation_mode=align_corners, '
              'can use scales(*) or sizes.\n'
              '\t5. mode=linear, coordinate_transformation_mode=asymmetric, '
              'can use scales(*) or sizes.\n'
              '\t6. mode=linear, coordinate_transformation_mode=half_pixel, '
              'can use scales(*) or sizes.\n'
              '\t7. mode=cubic, coordinate_transformation_mode=align_corners, '
              'cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) or '
              'sizes.\n'
              '\t8. mode=cubic, coordinate_transformation_mode=asymmetric, '
              'cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) or '
              'sizes.\n'
              '\t9. mode=cubic, coordinate_transformation_mode=half_pixel, '
              'cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) or '
              'sizes.\n'
              '\t10. mode=nearest, '
              'coordinate_transformation_mode=tf_crop_and_resize, '
              'extrapolation_value=any_float_value, '
              'nearest_mode=round_prefer_ceil, can use scales or sizes.\n'
              '\t11. mode=linear, '
              'coordinate_transformation_mode=tf_crop_and_resize, '
              'extrapolation_value=any_float_value, can use scales or sizes.\n'
              '\t- Note (*): The accuracy of your model will go down, if the '
              'height and the width of the new sizes(scales * origial sizes) '
              'are not in whole numbers.',
    'RoiAlign': 'sampling_ratio <= 0 is not fully supported.',
    'SplitToSequence': 'Scalar as the split input not supported.',
    'Upsample': 'Upsample required 4D input in Tensorflow.'
}
