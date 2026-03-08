import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, BayesianRidge
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedFeatureEngineer:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train = None
        self.test = None
        self.df = None  # 合并后的数据
        
    def load_data(self):
        print("正在加载数据...")
        self.train = pd.read_csv(self.train_path, sep=' ')
        self.test = pd.read_csv(self.test_path, sep=' ')
        print(f"训练集形状: {self.train.shape}")
        print(f"测试集形状: {self.test.shape}")
        
    def preprocess_step1_clean(self):
        """
        Step 1: 异常值处理与缺失值填充
        """
        print("\n[Step 1] 执行数据清洗与缺失值填充...")
        
        # 1. 异常值处理 (仅针对训练集)
        # 截断价格长尾 (保留 99% 的数据)
        quantile_99 = self.train['price'].quantile(0.99)
        print(f"价格 99% 分位数: {quantile_99}")
        self.train = self.train[self.train['price'] <= quantile_99]
        
        # 物理特征清洗 (power > 600 截断)
        self.train.loc[self.train['power'] > 600, 'power'] = 600
        self.test.loc[self.test['power'] > 600, 'power'] = 600
        
        # 合并数据以便统一处理
        self.train['is_train'] = 1
        self.test['is_train'] = 0
        self.df = pd.concat([self.train, self.test], ignore_index=True)
        
        # 处理 notRepairedDamage
        self.df['notRepairedDamage'] = self.df['notRepairedDamage'].replace('-', np.nan).astype(float)
        
        # 2. 缺失值精细化填充
        # bodyType: 同 model 的众数
        print("正在填充 bodyType...")
        self.df['bodyType'] = self.df.groupby('model')['bodyType'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median()))
        
        # fuelType: 同 model + brand 的众数
        print("正在填充 fuelType...")
        # 注意：groupby 多个键时，若某个组合全为空，则可能填不上，需回退到仅 model 或全局众数
        def fill_mode(x):
            return x.fillna(x.mode()[0] if not x.mode().empty else np.nan)
            
        self.df['fuelType'] = self.df.groupby(['brand', 'model'])['fuelType'].transform(fill_mode)
        self.df['fuelType'] = self.df.groupby('brand')['fuelType'].transform(fill_mode)
        self.df['fuelType'] = self.df['fuelType'].fillna(self.df['fuelType'].mode()[0])
        
        # gearbox: 同 model 的众数
        print("正在填充 gearbox...")
        self.df['gearbox'] = self.df.groupby('model')['gearbox'].transform(fill_mode)
        self.df['gearbox'] = self.df['gearbox'].fillna(self.df['gearbox'].mode()[0])
        
        # power: 同 brand + bodyType 的均值
        print("正在填充 power...")
        self.df['power'] = self.df.groupby(['brand', 'bodyType'])['power'].transform(lambda x: x.fillna(x.mean()))
        self.df['power'] = self.df['power'].fillna(self.df['power'].mean())
        
        print("Step 1 完成。")

    def preprocess_step2_business_features(self):
        """
        Step 2: 业务逻辑特征构造
        """
        print("\n[Step 2] 构造业务逻辑特征...")
        
        # 1. 精准车龄
        # 解析 YYYYMMDD
        # 有些日期可能是非法的 (例如月份是00)，需要处理
        # 这里简单起见，提取年份和月份计算
        def parse_date(x):
            s = str(int(x))
            year = int(s[:4])
            month = int(s[4:6])
            day = int(s[6:8])
            return year, month, day

        # 由于数据量大，直接向量化处理
        # 处理 regDate (注册日期)
        # 发现有些 regDate 的月份是 00，这通常是异常，我们将其视为 01 月
        self.df['regDate_year'] = (self.df['regDate'] // 10000).astype(int)
        self.df['regDate_month'] = ((self.df['regDate'] // 100) % 100).astype(int)
        # 修正异常月份
        self.df.loc[self.df['regDate_month'] == 0, 'regDate_month'] = 1
        
        # 处理 creatDate (上线日期)
        self.df['creatDate_year'] = (self.df['creatDate'] // 10000).astype(int)
        self.df['creatDate_month'] = ((self.df['creatDate'] // 100) % 100).astype(int)
        
        # 计算车龄 (以天为单位太细，以月为单位，或年小数)
        self.df['car_age_years'] = (self.df['creatDate_year'] - self.df['regDate_year']) + \
                                   (self.df['creatDate_month'] - self.df['regDate_month']) / 12.0
        
        # 修正可能的负数车龄 (数据噪声)
        self.df.loc[self.df['car_age_years'] < 0, 'car_age_years'] = 0
        
        # 2. 折旧率特征 miles_per_year
        self.df['miles_per_year'] = self.df['kilometer'] / (self.df['car_age_years'] + 1) # +1 防止除零
        
        # 3. 动力里程比
        self.df['power_per_km'] = self.df['power'] / (self.df['kilometer'] + 1)
        
        # 4. 上线时间特征 (季节性)
        self.df['creat_month'] = self.df['creatDate_month']
        
        # 5. Kilometer 分箱
        self.df['kilometer_bin'] = pd.cut(self.df['kilometer'], 5, labels=False)
        
        print("Step 2 完成。")

    def preprocess_step3_target_encoding(self):
        """
        Step 3: 5-Fold Target Encoding
        """
        print("\n[Step 3] 执行 5-Fold Target Encoding...")
        
        # 需要编码的类别特征
        cat_cols = ['brand', 'model', 'bodyType', 'fuelType', 'gearbox', 'regionCode', 'kilometer_bin']
        
        # 仅在训练集上计算统计量，映射到测试集，防止泄露
        # 但为了更稳健，我们使用 K-Fold 方式给训练集生成特征，测试集使用全量训练集的统计量
        
        train_df = self.df[self.df['is_train'] == 1].copy()
        test_df = self.df[self.df['is_train'] == 0].copy()
        
        # 必须重置索引，否则 kfold 会报错
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
        # 目标变量 (先做 log1p 变换)
        y_train = np.log1p(train_df['price'])
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in cat_cols:
            # 初始化新特征列
            train_df[f'{col}_target_mean'] = 0
            test_df[f'{col}_target_mean'] = 0
            
            # 1. 训练集 K-Fold 编码
            for train_index, val_index in kf.split(train_df):
                X_tr, X_val = train_df.iloc[train_index], train_df.iloc[val_index]
                y_tr = y_train.iloc[train_index]
                
                # 计算均值
                feat_mean = X_tr.groupby(col).apply(lambda x: y_tr.loc[x.index].mean())
                
                # 映射到验证集
                train_df.loc[val_index, f'{col}_target_mean'] = X_val[col].map(feat_mean)
            
            # 填充 K-Fold 产生的 NaN (未出现的类别用全局均值)
            global_mean = y_train.mean()
            train_df[f'{col}_target_mean'].fillna(global_mean, inplace=True)
            
            # 2. 测试集编码 (使用全量训练集统计)
            feat_mean_full = train_df.groupby(col).apply(lambda x: y_train.loc[x.index].mean())
            test_df[f'{col}_target_mean'] = test_df[col].map(feat_mean_full)
            test_df[f'{col}_target_mean'].fillna(global_mean, inplace=True)
            
        # 更新 self.df
        self.df = pd.concat([train_df, test_df], ignore_index=True)
        
        # 简单的 Label Encoding (给树模型用原始类别)
        for col in cat_cols:
             le = LabelEncoder()
             self.df[col] = le.fit_transform(self.df[col].astype(str))
             
        # 额外的匿名特征交互 (简单版)
        self.df['v0_v12_prod'] = self.df['v_0'] * self.df['v_12']
        self.df['v0_v3_sum'] = self.df['v_0'] + self.df['v_3']
             
        # 最终缺失值检查与填充
        print("执行最终缺失值填充 (剩余缺失值填 -1)...")
        self.df = self.df.fillna(-1)
        
        print("Step 3 完成。")
        
    def get_data(self):
        train_data = self.df[self.df['is_train'] == 1].drop(['is_train'], axis=1)
        test_data = self.df[self.df['is_train'] == 0].drop(['is_train'], axis=1) # 测试集包含 price 列 (虽然是 NaN)
        return train_data, test_data

def train_evaluate_extra_trees(train_df):
    print("\n[Verification] 训练 Extra Trees 验证性能...")
    
    # 准备特征和目标
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'price', 
                 'regDate_year', 'regDate_month', 'creatDate_year', 'creatDate_month'] 
    
    features = [c for c in train_df.columns if c not in drop_cols]
    target = np.log1p(train_df['price']) # 使用 Log1p 目标
    
    X = train_df[features]
    y = target
    
    # 使用 5-Fold CV 验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42, min_samples_split=2)
        model.fit(X_train, y_train)
        
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log) # 还原
        y_val_orig = np.expm1(y_val)
        
        mae = mean_absolute_error(y_val_orig, y_pred)
        mae_scores.append(mae)
        oof_preds[val_idx] = y_pred_log # 存储 log 预测值
        
        print(f"Fold {fold+1} MAE: {mae:.4f}")
        
    avg_mae = np.mean(mae_scores)
    print(f"\nAverage MAE (5-Fold): {avg_mae:.4f}")
    
    # 特征重要性 (使用最后一折的模型)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances:")
    for i in range(10):
        print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")
        
    return avg_mae, oof_preds, features

if __name__ == "__main__":
    train_path = r'd:\Case-二手车价格预测\used_car_train_20200313.csv'
    test_path = r'd:\Case-二手车价格预测\used_car_testB_20200421.csv'
    
    engineer = AdvancedFeatureEngineer(train_path, test_path)
    engineer.load_data()
    engineer.preprocess_step1_clean()
    engineer.preprocess_step2_business_features()
    engineer.preprocess_step3_target_encoding()
    
    train_data, test_data = engineer.get_data()
    
    # 步骤 4: Stacking 堆叠模型
    print("\n[Step 4] 构建 Stacking 框架...")
    
    # 准备数据
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'price', 
                 'regDate_year', 'regDate_month', 'creatDate_year', 'creatDate_month'] 
    features = [c for c in train_data.columns if c not in drop_cols]
    target = np.log1p(train_data['price'])
    
    X = train_data[features]
    y = target
    
    # 定义基模型
    lgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    models = {
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42, min_samples_split=2),
        'LightGBM': lgb.LGBMRegressor(**lgb_params),
        'XGBoost': xgb.XGBRegressor(**xgb_params),
        'RandomForest': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42) # 减少树数量以加快速度
    }
    
    # 5折 Stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储 OOF 预测结果 (作为第二层的特征)
    oof_train = pd.DataFrame()
    
    # 存储测试集预测结果
    test_preds_dict = {}
    
    # 评估各基模型
    print(f"基模型训练开始 (Features: {len(features)})...")
    
    for name, model in models.items():
        print(f"训练 {name}...")
        oof_preds = np.zeros(X.shape[0])
        mae_scores = []
        
        # 对测试集的预测 (5折平均)
        test_preds = np.zeros(test_data.shape[0])
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            
            # 验证集预测
            y_pred_log = model.predict(X_val)
            oof_preds[val_idx] = y_pred_log
            
            # 还原 MAE 计算
            y_pred = np.expm1(y_pred_log)
            y_val_orig = np.expm1(y_val)
            mae = mean_absolute_error(y_val_orig, y_pred)
            mae_scores.append(mae)
            
            # 测试集预测 (累加)
            test_preds += model.predict(test_data[features]) / kf.get_n_splits()
            
        print(f"  {name} 5-Fold CV MAE: {np.mean(mae_scores):.4f}")
        oof_train[name] = oof_preds
        test_preds_dict[name] = test_preds
        
    # 第二层: Ridge Regression
    print("\n训练 Meta Model (Ridge)...")
    meta_model = Ridge(alpha=10, random_state=42)
    
    meta_oof_preds = np.zeros(X.shape[0])
    meta_mae_scores = []
    
    # 对 Stacking 特征再次进行 CV (或者简单地在 OOF 上训练，但这可能过拟合，标准做法是再次 CV)
    # 但这里 OOF 已经是 Cross-Validated 的，所以可以直接作为特征训练 Meta Model?
    # 不，通常 Meta Model 也是 CV 训练的，或者直接在完整 OOF 上训练？
    # 标准 Stacking 流程:
    # 第一层: 交叉验证生成 OOF 预测
    # 第二层: 基于 OOF 预测训练最终模型
    
    # 这里我们再次使用 CV 评估 Meta Model 的性能
    for fold, (train_idx, val_idx) in enumerate(kf.split(oof_train, y)):
        X_meta_train, X_meta_val = oof_train.iloc[train_idx], oof_train.iloc[val_idx]
        y_meta_train, y_meta_val = y.iloc[train_idx], y.iloc[val_idx]
        
        meta_model.fit(X_meta_train, y_meta_train)
        
        y_pred_log = meta_model.predict(X_meta_val)
        meta_oof_preds[val_idx] = y_pred_log
        
        y_pred = np.expm1(y_pred_log)
        y_val_orig = np.expm1(y_meta_val)
        mae = mean_absolute_error(y_val_orig, y_pred)
        meta_mae_scores.append(mae)
        
    final_mae = np.mean(meta_mae_scores)
    print(f"\nStacking Model Final MAE: {final_mae:.4f}")
    
    if final_mae < 500:
        print("SUCCESS: 达成 MAE < 500 目标！")
    else:
        print(f"WARNING: 未达成目标 (差 {final_mae - 500:.4f})")
        
    # 生成提交文件
    print("\n正在生成提交文件...")
    
    # 重新在全量 OOF 上训练 Meta Model
    meta_model.fit(oof_train, y)
    
    # 准备测试集 Meta Features
    test_meta_features = pd.DataFrame(test_preds_dict)
    # 确保列顺序一致
    test_meta_features = test_meta_features[oof_train.columns]
    
    # 预测测试集
    test_pred_log = meta_model.predict(test_meta_features)
    test_pred = np.expm1(test_pred_log)
    
    # 读取 Sample Submission 以获取 SaleID
    # 但我们已经在 load_data 中读取了 test，可以从那里获取 SaleID (如果之前没 drop 的话)
    # 这里的 test_data 已经 drop 了 SaleID，所以需要重新读取或者从 engineer.test 中获取
    
    # engineer.test 包含原始测试集数据 (含 SaleID)
    submission = pd.DataFrame()
    submission['SaleID'] = engineer.test['SaleID']
    submission['price'] = test_pred
    
    # 简单的后处理：价格不能为负
    submission.loc[submission['price'] < 0, 'price'] = 0
    
    output_dir = r'd:\Case-二手车价格预测\Advanced_Stacking_Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sub_path = os.path.join(output_dir, f'submission_stacking_mae{final_mae:.2f}.csv')
    submission.to_csv(sub_path, index=False)
    print(f"提交文件已保存至: {sub_path}")
    
    # 保存结果报告
    report_path = os.path.join(output_dir, 'model_performance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Advanced Stacking Model Performance Report\n")
        f.write("========================================\n")
        f.write(f"Final Stacking MAE: {final_mae:.4f}\n\n")
        f.write("Base Models 5-Fold CV MAE:\n")
        f.write("-" * 30 + "\n")
        for name in models.keys():
            # 这里我们无法直接获取之前 loop 中的 mae_scores，只能打印
            # 改进：在 loop 中存储 scores
            pass
        f.write("(See console output for details)\n") 
        
    print("任务全部完成。")

