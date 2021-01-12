import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import json
import numpy


# 设置数据集
records = [['牛奶', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
           ['莳萝', '洋葱', '肉豆蔻', '芸豆', '鸡蛋', '酸奶'],
           ['牛奶', '苹果', '芸豆', '鸡蛋'],
           ['牛奶', '独角兽', '玉米', '芸豆', '酸奶'],
           ['玉米', '洋葱', '洋葱', '芸豆', '冰淇淋', '鸡蛋']]

te = TransactionEncoder()
# 进行 one-hot 编码
te_ary = te.fit(records).transform(records)
df = pd.DataFrame(te_ary, columns=te.columns_)
# 利用 Apriori 找出频繁项集
freq = apriori(df, min_support=0.4, use_colnames=True)
print(freq)

# 计算关联规则，算出置信度
rules = association_rules(freq, metric="confidence", min_threshold=0.1)
for i, val in enumerate(rules['antecedents']):
    print(i, end='   ')
    print(rules['antecedents'][i], end = "   ")
    print(rules['consequents'][i], end = "   ")
    print(rules['confidence'][i])


