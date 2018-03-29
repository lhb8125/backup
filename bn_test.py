import numpy as np
import mxnet as mx

def get_symbol():
    data = mx.symbol.var('data')

    bn1 = mx.symbol.BatchNorm_v1(data=data, eps=1e-5, fix_gamma=False, use_global_stats=False)
    return mx.symbol.Convolution(data=bn1,num_filter=3,kernel=(3,3),stride=(1,1),pad=(1,1),no_bias=True,name='conv1',workspace=512)
#    return mx.symbol.BatchNorm_v1(data=conv1, eps=1e-5, fix_gamma=False, use_global_stats=False)

def main():
    symbol = get_symbol()
    module = mx.mod.Module(symbol, context=[mx.gpu(0)])
    batch_size = 3
    module.bind([('data', (batch_size, 3, 10, 10))])
    module.init_params()
    module.init_optimizer()
    input_data = np.zeros((batch_size, 3, 10, 10))
    for i in range(batch_size):
        input_data[i] = i
    data_batch = mx.io.DataBatch([mx.nd.array(input_data)])
    module.forward(data_batch)
    output = module.get_outputs()[0].asnumpy()
    print output

if __name__ == '__main__':
    main()
