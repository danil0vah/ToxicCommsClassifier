import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path: str, test_size: float = 0.2) -> list:
    df = pd.read_csv(file_path)
    df['toxic'] = df['toxic'].astype(int)
    
    return train_test_split(df, test_size=test_size)