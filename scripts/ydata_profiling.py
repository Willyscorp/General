from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np

def random_sample(data, sample_size=5, random_state=None, replace=False):
   
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.sample(n=sample_size if isinstance(sample_size, int) else None,
                           frac=sample_size if isinstance(sample_size, float) else None,
                           random_state=random_state,
                           replace=replace)
    
    elif isinstance(data, np.ndarray):
        if isinstance(sample_size, float):
            sample_size = int(len(data) * sample_size)
        np.random.seed(random_state)
        indices = np.random.choice(len(data), size=sample_size, replace=replace)
        return data[indices]
    
    else:
        raise TypeError("Data must be a pandas DataFrame, Series, or numpy array.")

df = pd.read_csv("data/raw/life_style_data.csv")
sample_df = random_sample(df, sample_size=0.05, random_state=10)

profile = ProfileReport(sample_df, title="Lifestyle Data Full Profile Sample", explorative=True)
profile.to_file("analysis_outputs/lifestyle_full_profile_sample_1.html")
