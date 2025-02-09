import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

# Veri setinin okunması
data = pd.read_csv("C:/Users/Burak/Desktop/General/Projects/Introduction to Pattern Recognition -1/PatternProject-1/otu/otu.csv", low_memory=False)

# Veri setinin sağ el ve sol el verilerine ayrılması
left_hand_data = data.loc[:, data.iloc[0, :] == 'left']  # Sol el
right_hand_data = data.loc[:, data.iloc[0, :] == 'right']  # Sağ el

# Sol el veri setinin özellikler ve hedef değişkenler olarak ikiye ayrılması
left_hand_features = left_hand_data.iloc[1:, :].T  # Özelliklerin sütunlara yerleştirilmesi
left_hand_target = left_hand_data.iloc[0, :].values  # Hedef etiketlerin belirlenmesi

# Sağ el veri setinin özellikler ve hedef değişkenler olarak ikiye ayrılması
right_hand_features = right_hand_data.iloc[1:, :].T  # Özelliklerin sütunlara yerleştirilmesi
right_hand_target = right_hand_data.iloc[0, :].values  # Hedef etiketlerin belirlenmesi

# Boyutların eşitlenmesi
right_hand_target = right_hand_target[:left_hand_target.shape[0]]
right_hand_features = right_hand_features[:left_hand_features.shape[0]]

# Verilerin birleştirilmesi
features = pd.concat([left_hand_features, right_hand_features], axis=0)
target = pd.concat([pd.Series(left_hand_target), pd.Series(right_hand_target)], ignore_index=True)

# Decision Tree algoritmasının tanımlanması
model_DT = DecisionTreeClassifier()

# En iyi 100 özniteliğin seçimi için Decision Tree'nin kullanılması
selector = SelectFromModel(estimator=model_DT, max_features=100)
selected_features = selector.fit_transform(features, target)

# Cross-Validation Accuracy değerinin hesaplanması ve yazdırılması
cv_results = cross_val_score(model_DT, selected_features, target, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_results.mean())

# Random Forest algoritmasının tanımlanması
model_RFE = RandomForestClassifier()

# Training ve Test setlerinin belirlenmesi
X_train, X_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.2, random_state=42)

# Training set ile modelin eğitilmesi
model_RFE.fit(X_train, y_train)

# Test seti üzerinden tahmin üretilmesi
predictions = model_RFE.predict(X_test)

# Classification Accuracy değerinin hesaplanması ve yazdırılması
accuracy = accuracy_score(y_test, predictions)
print("Classification Accuracy:", accuracy)

# Confusion Matrix'in oluşturulması
conf_matrix = confusion_matrix(y_test, predictions)
TN = conf_matrix[0][0]  # True Negative
FP = conf_matrix[0][1]  # False Positive
FN = conf_matrix[1][0]  # False Negative
TP = conf_matrix[1][1]  # True Positive

# Sensitivity ve Specificity değerlerinin hesaplanması
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# Değerlerin yazdırılması
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Confusion Matrix:")
print(conf_matrix)

# AUC değerinin hesaplanması ve yazdırılması
probabilities = model_RFE.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probabilities)
print("AUC:", auc)