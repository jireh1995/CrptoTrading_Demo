from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data):
    feature_names = ['Open', 'High', 'Low', 'Volume']
    X = data[feature_names].values  # 使用 .values 只保留数据
    y = (data['Close'] > data['Open']).astype(int).values  # 使用 .values 只保留数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf, feature_names
