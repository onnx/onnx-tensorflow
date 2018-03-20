ONNX-Tensorflow Support Status
======

Backend
______

| ONNX Op        | Supported Version  |
| -------------- |:------------------:|
|unsqueeze|1|
|arg_min|1|
|less|1|
|lp_normalization|1|
|shape|1, 2, 3, 4, 5|
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
|tanh|1, 2, 3, 4, 5|
|greater|1|
|ceil|1, 2, 3, 4, 5|
|leaky_relu|1|
|not|1, 2, 3, 4, 5|
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
|softsign|1, 2, 3, 4, 5|
|crop|N/A|
|flatten|1|
|reduce_sum_square|1|
|scale|N/A|
|xor|1|
|sub|1|
|sigmoid|1, 2, 3, 4, 5|
|neg|1, 2, 3, 4, 5|
|sum|1|
|abs|1, 2, 3, 4, 5|
|max|1|
|loop_index_tensor|N/A|
|mul|1|
|random_uniform|1, 2, 3, 4, 5|
|softmax|1|
|top_k|1|
|max_pool|1|
|global_average_pool|1|
|transpose|1, 2, 3, 4, 5|
|average_pool|1|
|upsample|N/A|
|conv|1|
|equal|1|
|hard_sigmoid|1|
|softplus|1, 2, 3, 4, 5|
|random_normal_like|1|
|arg_max|1|
|or|1|
|scaled_tanh|N/A|
|pow|1|
|size|1, 2, 3, 4, 5|
|log|1, 2, 3, 4, 5|
|relu|1, 2, 3, 4, 5|
|random_normal|1, 2, 3, 4, 5|
|concat|1, 4|
|mean_variance_normalization|N/A|
|a_ten|N/A|
|random_uniform_like|1|
|dropout|1|
|global_max_pool|1|
|squeeze|1, 2, 3, 4, 5|
|reduce_log_sum_exp|1, 2, 3, 4, 5|
|reduce_mean|1, 2, 3, 4, 5|
|reciprocal|1, 2, 3, 4, 5|
|f_c|N/A|
|l_s_t_m|1|
|cast|1, 2, 3, 4, 5|
|loop|N/A|
|and|1|
|constant|1|
|clip|1|
|lp_pool|N/A|
|g_r_u_unit|N/A|
|max_roi_pool|N/A|
|if|N/A|
|reduce_prod|1, 2, 3, 4, 5|
|floor|1, 2, 3, 4, 5|
|reshape|1, 5|
|sqrt|1, 2, 3, 4, 5|
|reduce_max|1, 2, 3, 4, 5|
|pad|1, 2|
|split|1, 2|
|tile|1|
|selu|1|
|log_softmax|1|
|g_r_u|N/A|
|reduce_sum|1, 2, 3, 4, 5|
|l_r_n|1|
|reduce_min|1, 2, 3, 4, 5|
|slice|1|
|constant_fill|N/A|
|identity|1, 2, 3, 4, 5|
|gather|1, 2, 3, 4, 5|
|space_to_depth|1|
|exp|1, 2, 3, 4, 5|
|p_relu|1|


Frontend
______

| ONNX Op        | Supported Version  |
| -------------- |:------------------:|
|unsqueeze|N/A|
|arg_min|N/A|
|less|N/A|
|lp_normalization|N/A|
|shape|N/A|
|hardmax|N/A|
|thresholded_relu|N/A|
|affine|N/A|
|depth_to_space|N/A|
|instance_normalization|N/A|
|add|1, 2, 3, 4, 5|
|reduce_l1|N/A|
|reduce_l2|N/A|
|batch_normalization|N/A|
|gemm|N/A|
|elu|N/A|
|tanh|1, 2, 3, 4, 5|
|greater|N/A|
|ceil|N/A|
|leaky_relu|N/A|
|not|1, 2, 3, 4, 5|
|image_scaler|N/A|
|reduce_log_sum|N/A|
|mat_mul|N/A|
|r_n_n|N/A|
|conv_transpose|N/A|
|global_lp_pool|N/A|
|min|N/A|
|div|N/A|
|mean|N/A|
|given_tensor_fill|N/A|
|parametric_softplus|N/A|
|softsign|N/A|
|crop|N/A|
|flatten|N/A|
|reduce_sum_square|N/A|
|scale|N/A|
|xor|1|
|sub|1|
|sigmoid|1, 2, 3, 4, 5|
|neg|N/A|
|sum|N/A|
|abs|N/A|
|max|N/A|
|loop_index_tensor|N/A|
|mul|1, 2, 3, 4, 5|
|random_uniform|1|
|softmax|N/A|
|top_k|N/A|
|max_pool|N/A|
|global_average_pool|N/A|
|transpose|1|
|average_pool|N/A|
|upsample|N/A|
|conv|N/A|
|equal|N/A|
|hard_sigmoid|N/A|
|softplus|N/A|
|random_normal_like|N/A|
|arg_max|N/A|
|or|1|
|scaled_tanh|N/A|
|pow|1, 2, 3, 4, 5|
|size|N/A|
|log|N/A|
|relu|1, 2, 3, 4, 5|
|random_normal|1|
|concat|1, 4|
|mean_variance_normalization|N/A|
|a_ten|N/A|
|random_uniform_like|N/A|
|dropout|N/A|
|global_max_pool|N/A|
|squeeze|1|
|reduce_log_sum_exp|N/A|
|reduce_mean|1|
|reciprocal|1, 2, 3, 4, 5|
|f_c|N/A|
|l_s_t_m|N/A|
|cast|N/A|
|loop|N/A|
|and|1|
|constant|N/A|
|clip|N/A|
|lp_pool|N/A|
|g_r_u_unit|N/A|
|max_roi_pool|N/A|
|if|N/A|
|reduce_prod|1|
|floor|N/A|
|reshape|1, 5|
|sqrt|1, 2, 3, 4, 5|
|reduce_max|1|
|pad|1, 2|
|split|1, 2|
|tile|N/A|
|selu|N/A|
|log_softmax|N/A|
|g_r_u|N/A|
|reduce_sum|1|
|l_r_n|N/A|
|reduce_min|1|
|slice|N/A|
|constant_fill|N/A|
|identity|N/A|
|gather|N/A|
|space_to_depth|N/A|
|exp|N/A|
|p_relu|N/A|
