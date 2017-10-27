"""
预处理常用工具类
"""

import pandas as pd


# 获取某一类的特征列名
def get_column_name_list_by_prefix(df: pd.DataFrame, prefix: str) -> list:
    columns = list(df.columns)
    return [col for col in columns if col.startswith(prefix)]


# 检查特征列里有没有缺失值
def get_column_name_list_with_missing_value(df: pd.DataFrame, feature: list, placeholder) -> list:
    col_has_missing_value = []
    for col in feature:
        value_counts = df[col].value_counts()
        value_counts_index = list(value_counts.index)
        if placeholder in value_counts_index:
            col_has_missing_value.append(col)
    return col_has_missing_value


def get_column_type_pair_list(df: pd.DataFrame, prefix: str=None) -> list:
    dtype_info = df.dtypes
    column_type_pair_list = list(zip(list(dtype_info.index), [col.name for col in dtype_info]))
    if prefix is not None:
        column_type_pair_list = list(filter(lambda x: str(x[0]).startswith(prefix), column_type_pair_list))
    return column_type_pair_list


def fix_feature_ps_ind(df: pd.DataFrame) -> pd.DataFrame:
    # 存在缺失值的都是 category, 直接 one-hot编码
    df_columns = list(df.columns)
    _df = pd.get_dummies(
        data=df,
        columns=[col for col in df_columns if '_cat' in col]
    )

    return _df


# 给每个列加上一个基数
def indexing(df: pd.DataFrame, columns: list, offset: int = 0, neat: bool = True) -> (pd.DataFrame, int):
    df = df.copy()

    for col in columns:
        # 如果要精简, 每列先减去其最小值

        col_min = df[col].min()
        col_max = df[col].max()

        col_value_range_length = col_max - col_min + 1

        # 每列减去最小值
        if col_min != 0 and neat:
            df[col] = df[col] - col_min

        col_unique_value = sorted(pd.unique(df[col]))
        col_unique_value_number = \
            len(col_unique_value) if col_value_range_length <= len(col_unique_value) else col_value_range_length

        print(col, col_max, col_min, col_unique_value_number, col_unique_value)

        df[col] = df[col] + offset
        offset = offset + col_unique_value_number

    return df, offset

