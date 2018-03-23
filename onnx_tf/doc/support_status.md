ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|a_ten|N/A|
|abs|1|
|add|1|
|affine|N/A|
|and|1|
|arg_max|1|
|arg_min|1|
|average_pool|1|
|batch_normalization|1, 6|
|cast|1|
|ceil|1|
|clip|1|
|concat|1, 4|
|constant|1|
|constant_fill|N/A|
|conv|1|
|conv_transpose|1|
|crop|N/A|
|depth_to_space|1|
|div|1|
|dropout|1|
|elu|1|
|equal|1|
|exp|1|
|f_c|N/A|
|flatten|1|
|floor|1|
|g_r_u|N/A|
|g_r_u_unit|N/A|
|gather|1|
|gemm|1|
|given_tensor_fill|N/A|
|global_average_pool|1|
|global_lp_pool|1, 2|
|global_max_pool|1|
|greater|1|
|hard_sigmoid|1|
|hardmax|1|
|identity|1|
|if|N/A|
|image_scaler|N/A|
|instance_normalization|N/A|
|l_r_n|1|
|l_s_t_m|1|
|leaky_relu|1|
|less|1|
|log|1|
|log_softmax|1|
|loop|N/A|
|loop_index_tensor|N/A|
|lp_normalization|1|
|lp_pool|N/A|
|mat_mul|1|
|max|1|
|max_pool|1|
|max_roi_pool|N/A|
|mean|1|
|mean_variance_normalization|N/A|
|min|1|
|mul|1|
|neg|1|
|not|1|
|or|1|
|p_relu|1|
|pad|1, 2|
|parametric_softplus|N/A|
|pow|1|
|r_n_n|N/A|
|random_normal|1|
|random_normal_like|1|
|random_uniform|1|
|random_uniform_like|1|
|reciprocal|1|
|reduce_l1|1|
|reduce_l2|N/A|
|reduce_log_sum|N/A|
|reduce_log_sum_exp|1|
|reduce_max|1|
|reduce_mean|1|
|reduce_min|1|
|reduce_prod|1|
|reduce_sum|1|
|reduce_sum_square|1|
|relu|1|
|reshape|1, 5|
|scale|N/A|
|scaled_tanh|N/A|
|selu|1|
|shape|1|
|sigmoid|1|
|size|1|
|slice|1|
|softmax|1|
|softplus|1|
|softsign|1|
|space_to_depth|1|
|split|1, 2|
|sqrt|1|
|squeeze|1|
|sub|1|
|sum|1|
|tanh|1|
|thresholded_relu|1|
|tile|1|
|top_k|1|
|transpose|1|
|unsqueeze|1|
|upsample|N/A|
|xor|1|


Frontend
______

| Tensorflow Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|add|1|
|avg_pool|1|
|bias_add|1|
|concat_v2|1, 4|
|conv1_d|1|
|conv2_d|1|
|conv3_d|1|
|identity|1||
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
|reciprocal|1|
|relu|1|
|reshape|1, 5|
|sigmoid|1|
|softmax|1|
|split_v|1, 2|
|sqrt|1|
|squeeze|1|
|sub|1|
|sum|1|
|tanh|1|
|transpose|1|
