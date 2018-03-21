ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|unsqueeze|1|
|arg_min|1|
|less|1|
|lp_normalization|1|
|shape|1|
|hardmax|1|
|thresholded_relu|1|
|affine|N/A|
|depth_to_space|1|
|instance_normalization|N/A|
|add|1|
|reduce_l1|1|
|reduce_l2|N/A|
|batch_normalization|1|
|gemm|1|
|elu|1|
|tanh|1|
|greater|1|
|ceil|1|
|leaky_relu|1|
|not|1|
|image_scaler|N/A|
|reduce_log_sum|N/A|
|mat_mul|1|
|r_n_n|N/A|
|conv_transpose|1|
|global_lp_pool|1, 2|
|min|1|
|div|1|
|mean|1|
|given_tensor_fill|N/A|
|parametric_softplus|N/A|
|softsign|1|
|crop|N/A|
|flatten|1|
|reduce_sum_square|1|
|scale|N/A|
|xor|1|
|sub|1|
|sigmoid|1|
|neg|1|
|sum|1|
|abs|1|
|max|1|
|loop_index_tensor|N/A|
|mul|1|
|random_uniform|1|
|softmax|1|
|top_k|1|
|max_pool|1|
|global_average_pool|1|
|transpose|1|
|average_pool|1|
|upsample|N/A|
|conv|1|
|equal|1|
|hard_sigmoid|1|
|softplus|1|
|random_normal_like|1|
|arg_max|1|
|or|1|
|scaled_tanh|N/A|
|pow|1|
|size|1|
|log|1|
|relu|1|
|random_normal|1|
|concat|1, 4|
|mean_variance_normalization|N/A|
|a_ten|N/A|
|random_uniform_like|1|
|dropout|1|
|global_max_pool|1|
|squeeze|1|
|reduce_log_sum_exp|1|
|reduce_mean|1|
|reciprocal|1|
|f_c|N/A|
|l_s_t_m|1|
|cast|1|
|loop|N/A|
|and|1|
|constant|1|
|clip|1|
|lp_pool|N/A|
|g_r_u_unit|N/A|
|max_roi_pool|N/A|
|if|N/A|
|reduce_prod|1|
|floor|1|
|reshape|1, 5|
|sqrt|1|
|reduce_max|1|
|pad|1, 2|
|split|1, 2|
|tile|1|
|selu|1|
|log_softmax|1|
|g_r_u|N/A|
|reduce_sum|1|
|l_r_n|1|
|reduce_min|1|
|slice|1|
|constant_fill|N/A|
|identity|1|
|gather|1|
|space_to_depth|1|
|exp|1|
|p_relu|1|


Frontend
______

| Tensorflow Op        | Supported ONNX Version  |
| -------------- |:------------------:|
|random_standard_normal|1|
|sigmoid|1|
|pow|1|
|logical_or|1|
|logical_and|1|
|sqrt|1|
|logical_xor|1|
|sub|1|
|min|1|
|reshape|1, 5|
|sum|1|
|relu|1|
|add|1|
|pad|1, 2|
|mul|1|
|prod|1|
|random_uniform|1|
|max|1|
|transpose|1|
|squeeze|1|
|split_v|1, 2|
|concat_v2|1, 4|
|logical_not|1|
|reciprocal|1|
|tanh|1|
|mean|1|
