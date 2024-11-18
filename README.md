# Data-mine-fall-2024
dataset link
crime :https://hub.mph.in.gov/dataset/indiana-arrest-data
![image](https://github.com/user-attachments/assets/b15c0ea9-2618-4ce8-b0a3-929d048851b9)

ND:https://www.ngdc.noaa.gov/hazard/hazards.shtml
![image](https://github.com/user-attachments/assets/c7cce776-778b-4193-bc3a-cef0664a74a6)

model 1:
Used prophet lib for forcasting crime counts  
Tuned Prophet MAE: 5729.2685
Tuned Prophet RMSE: 6708.8788
model 2:
Used ElasticNet with Bayesian optimization for ElasticNet and holidays lib for holidays season
ElasticNet with Seasonality - MAE: 1825.98, RMSE: 2436.17, R²: 0.90, MAPE: 2.76%
Best ElasticNet Parameters: OrderedDict([('alpha', 0.1), ('l1_ratio', 0.9)])
