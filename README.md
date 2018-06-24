本项目为为Rossmann Store Sales建立销售预测模型
===

模型建立分为5个部分<br>
  1.数据探索<br>
该部分参考Omar El GabryA(https://www.kaggle.com/omarelgabry/a-journey-through-rossmann-stores) 的内容进行数据探索

  2.算法比较<br>
通过对LinearRegression，RandomForestRegressor，DecisionTreeRegressor，XGboost四个算法比较进行算法性能比较

  3.算法选择<br>
通过对RandomForestRegressor，XGboost算法进行参数调优后进行比较，最终选择xgboost

  4.XGboost算法调优<br>
在全量数据上对xgboost各个参数进行调优

  5.最终模型<br>
根据最优参数来对测试数据进行预测
