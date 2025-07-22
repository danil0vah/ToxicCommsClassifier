from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

def build_pipeline(tokenizer_fn) -> Pipeline:
    return Pipeline ([
        ('vertorizer', TfidfVectorizer(
            tokenizer=tokenizer_fn,
            max_df=0.9,
            min_df=3,
            sublinear_tf=True,
            norm='l2'
        )),
        ("model", GridSearchCV(
            LogisticRegression(random_state=0, class_weight='balanced'),
            param_grid={"C": [0.1, 1, 10]},
            cv=3,
            scoring='precision',
            verbose=4
        ))
    ])