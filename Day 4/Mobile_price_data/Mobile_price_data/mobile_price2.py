import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv(r'D:\ai work sapce\mobile_price_train.csv')
#print(df)
#print(df.corr())
correlation_matrix = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Mobile Features')
plt.show()
x=df[["ram",'battery_power',"px_height","px_width","price_range"]]
y=df["price_range"]     
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print ('xtest:\n',x_train)
# تطبيق الانحدار اللوجستي
model = LogisticRegression()
model.fit(x_train, y_train)

# التنبؤ بالنتائج
y_pred = model.predict(x_test)

r2=accuracy = accuracy_score(y_test, y_pred)
print(f"الدقة: {accuracy * 100:.2f}%")