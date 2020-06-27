backend_opset_version = {
    'Abs': [1, 6],
    'Acos': [7],
    'Acosh': [9],
    'Adagrad': [],
    'Adam': [],
    'Add': [1, 6, 7],
    'And': [1, 7],
    'ArgMax': [1, 11, 12],
    'ArgMin': [1, 11, 12],
    'ArrayFeatureExtractor': [],
    'Asin': [7],
    'Asinh': [9],
    'Atan': [7],
    'Atanh': [9],
    'AveragePool': [1, 7, 10, 11],
    'BatchNormalization': [1, 6, 7, 9],
    'Binarizer': [],
    'BitShift': [11],
    'Cast': [1, 6, 9],
    'CastMap': [],
    'CategoryMapper': [],
    'Ceil': [1, 6],
    'Celu': [],
    'Clip': [1, 6, 11, 12, 13],
    'Compress': [9, 11],
    'Concat': [1, 4, 11],
    'ConcatFromSequence': [11],
    'Constant': [1, 9, 11, 12],
    'ConstantFill': [1],
    'ConstantOfShape': [9],
    'Conv': [1, 11],
    'ConvInteger': [10],
    'ConvTranspose': [1, 11],
    'Cos': [7],
    'Cosh': [9],
    'CumSum': [11],
    'DepthToSpace': [1, 11],
    'DequantizeLinear': [10],
    'Det': [11],
    'DictVectorizer': [],
    'Div': [1, 6, 7],
    'Dropout': [1, 6, 7, 10],
    'DynamicQuantizeLinear': [11],
    'Einsum': [],
    'Elu': [1, 6],
    'Equal': [1, 7, 11],
    'Erf': [9],
    'Exp': [1, 6],
    'Expand': [8],
    'EyeLike': [9],
    'FeatureVectorizer': [],
    'Flatten': [1, 9, 11],
    'Floor': [1, 6],
    'GRU': [1, 3, 7],
    'Gather': [1, 11],
    'GatherElements': [11],
    'GatherND': [11],
    'Gemm': [1, 6, 7, 9, 11],
    'GlobalAveragePool': [1],
    'GlobalLpPool': [1, 2],
    'GlobalMaxPool': [1],
    'Gradient': [],
    'GraphCall': [],
    'Greater': [1, 7, 9],
    'GreaterOrEqual': [],
    'HardSigmoid': [1, 6],
    'Hardmax': [1, 11],
    'Identity': [1],
    'If': [1, 11],
    'ImageScaler': [1],
    'Imputer': [],
    'InstanceNormalization': [1, 6],
    'IsInf': [10],
    'IsNaN': [9],
    'LRN': [1],
    'LSTM': [1, 7],
    'LabelEncoder': [],
    'LeakyRelu': [1, 6],
    'Less': [1, 7, 9],
    'LessOrEqual': [],
    'LinearClassifier': [],
    'LinearRegressor': [],
    'Log': [1, 6],
    'LogSoftmax': [1, 11],
    'Loop': [1, 11],
    'LpNormalization': [1],
    'LpPool': [1, 2, 11],
    'MatMul': [1, 9],
    'MatMulInteger': [10],
    'Max': [1, 6, 8, 12, 13],
    'MaxPool': [1, 8, 10, 11, 12],
    'MaxRoiPool': [],
    'MaxUnpool': [9, 11],
    'Mean': [1, 6, 8],
    'MeanVarianceNormalization': [1, 9],
    'Min': [1, 6, 8, 12, 13],
    'Mod': [10],
    'Momentum': [],
    'Mul': [1, 6, 7],
    'Multinomial': [],
    'Neg': [1, 6],
    'NegativeLogLikelihoodLoss': [],
    'NonMaxSuppression': [10, 11],
    'NonZero': [9],
    'Normalizer': [],
    'Not': [1],
    'OneHot': [9, 11],
    'OneHotEncoder': [],
    'Or': [1, 7],
    'PRelu': [1, 6, 7, 9],
    'Pad': [1, 2, 11],
    'Pow': [1, 7],
    'QLinearConv': [10],
    'QLinearMatMul': [10],
    'QuantizeLinear': [10],
    'RNN': [1, 7],
    'RandomNormal': [1],
    'RandomNormalLike': [1],
    'RandomUniform': [1],
    'RandomUniformLike': [1],
    'Range': [11],
    'Reciprocal': [1, 6],
    'ReduceL1': [1, 11],
    'ReduceL2': [1, 11],
    'ReduceLogSum': [1, 11],
    'ReduceLogSumExp': [1, 11],
    'ReduceMax': [1, 11],
    'ReduceMean': [1, 11],
    'ReduceMin': [1, 11],
    'ReduceProd': [1, 11],
    'ReduceSum': [1, 11],
    'ReduceSumSquare': [1, 11],
    'Relu': [1, 6],
    'Reshape': [1, 5],
    'Resize': [10, 11],
    'ReverseSequence': [10],
    'RoiAlign': [],
    'Round': [11],
    'SVMClassifier': [],
    'SVMRegressor': [],
    'Scaler': [],
    'Scan': [8, 9, 11],
    'Scatter': [9],
    'ScatterElements': [11],
    'ScatterND': [11],
    'Selu': [1, 6],
    'SequenceAt': [11],
    'SequenceConstruct': [11],
    'SequenceEmpty': [11],
    'SequenceErase': [11],
    'SequenceInsert': [11],
    'SequenceLength': [11],
    'Shape': [1],
    'Shrink': [9],
    'Sigmoid': [1, 6],
    'Sign': [9],
    'Sin': [7],
    'Sinh': [9],
    'Size': [1],
    'Slice': [1, 10, 11],
    'Softmax': [1, 11],
    'SoftmaxCrossEntropyLoss': [],
    'Softplus': [1],
    'Softsign': [1],
    'SpaceToDepth': [1],
    'Split': [1, 2, 11, 13],
    'SplitToSequence': [11],
    'Sqrt': [1, 6],
    'Squeeze': [1, 11],
    'StringNormalizer': [],
    'Sub': [1, 6, 7],
    'Sum': [1, 6, 8],
    'Tan': [7],
    'Tanh': [1, 6],
    'TfIdfVectorizer': [9],
    'ThresholdedRelu': [1, 10],
    'Tile': [1, 6],
    'TopK': [1, 10, 11],
    'Transpose': [1],
    'TreeEnsembleClassifier': [],
    'TreeEnsembleRegressor': [],
    'Unique': [],
    'Unsqueeze': [1, 11],
    'Upsample': [7, 9],
    'Where': [9],
    'Xor': [1, 7],
    'ZipMap': []
}

backend_partial_support = {
    'Cast': 'Cast string to float32/float64/int32/int64 are not supported in '
            'Tensorflow.',
    'Clip': 'Clip input in uint64 is not supported in Tensorflow.',
    'ConcatFromSequence': 'new_axis=1 not supported in Tensorflow.',
    'ConvTranspose': 'ConvTranspose with dilations != 1, or transposed '
                     'convolution for 4D or higher are not supported in '
                     'Tensorflow.',
    'CumSum': 'CumSum inputs in uint32/uint64 are not supported in Tensorflow.',
    'Equal': 'Equal inputs in uint16/uint32/uint64 are not supported in '
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
    'Mod': 'Mod Dividend or Divisor in int8/int16/uint8/uint16/uint32/uint64 '
           'are not supported in Tensorflow.',
    'OneHot': 'OneHot indices in '
              'uint16/uint32/uint64/int8/int16/float16/float/double, or OneHot '
              'depth in '
              'uint8/uint16/uint32/uint64/int8/int16/int64/float16/float/double '
              'are not supported in Tensorflow.',
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
    'SplitToSequence': 'Scalar as the split input not supported.',
    'Upsample': 'Upsample required 4D input in Tensorflow.'
}
