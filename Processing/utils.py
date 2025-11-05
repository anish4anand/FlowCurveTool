def apply_offset(df):
    df = df.copy()
    df.iloc[:, 1] -= df.iloc[:, 1].mean()
    return df

def temperature_compensation(df):
    df = df.copy()
    df.iloc[:, 1] *= 0.95
    return df
