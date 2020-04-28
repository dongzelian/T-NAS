from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_multi = namedtuple('Genotype', 'normal_bottom normal_concat_bottom reduce_bottom reduce_concat_bottom \
                                         normal_mid normal_concat_mid reduce_mid reduce_concat_mid \
                                         normal_top normal_concat_top')


'''
Searched by Dongze Lian (2019/3/12)
'''
# 5-way 1-shot
AUTO_MAML_2 = Genotype(
    normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2),
            ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 2)],
    normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1),
            ('skip_connect', 3), ('dil_conv_5x5', 2), ('skip_connect', 4), ('skip_connect', 3)],
    reduce_concat=range(2, 6))

# 5-way 5-shot
AUTO_MAML_3 = Genotype(
    normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1),
            ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('skip_connect', 4), ('dil_conv_3x3', 3)],
    normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2),
            ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1)],
    reduce_concat=range(2, 6))





