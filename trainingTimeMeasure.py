
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout




column_name = ['age','sex','cp','treshbps','chol','fbs','restecg','thalach','exang','oldpeak','slop','ca','thal','HeartDisease']
raw_data = pd.read_excel('./datasets/heart-disease.xlsx' , header = None, names = column_name)
print(raw_data.head())

# preprocessing
# ? 와 같은 문자열은 예외로 처리해야할것
# 수가 큰 경우는 scaling을 미리 해서 발산하는 경우를 막아야함
raw_data.info()
# pd - pandas에서 제공하는 데이터형태로 편리하게 정보 확인가능
raw_data.describe()

clean_data = raw_data.replace('?' , np.nan) # NaN - Not a Number - float형이지만 연산 결과가 항상 NaN으로 나옴

clean_data = clean_data.dropna()
clean_data.info()
keep = column_name.pop() # pop - list를 stack으로 쓸 때 (index를 주면 중간에서 뺄 수도 있음 - 괄호안에 아무것도 없으면 stack처럼 사용)
#마지막 column을 제거하고 keep에 리턴해줌
print(keep)
print(column_name)


training_data = clean_data[column_name]
target = clean_data[[keep]]
print(training_data)
print(target)


from sklearn.preprocessing import StandardScaler # sklearn 사이킷런
scaler = StandardScaler()
scaled_data = scaler.fit_transform(training_data)
scaled_data = pd.DataFrame(scaled_data,columns=column_name)

print(scaled_data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaled_data,target,test_size=0.30)
# cross validation
# 전체 data중에 30%를 떼어놓고 학습(70%로)
# 학습 후 30%로 검증

# 문제를 다 외워버릴 위험성
# 데이터가 많이 없을 땐 어쩔 수 없음
# 훈련 데이터는 균형이 맞는것이 좋음.
# 검증 데이터는 조금 불균형해도 문제없다.


print('x_train shape', x_train.shape)
print('y_train shape',y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)
model = Sequential()
model.add(Dense(512,input_dim = 13, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0,2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0,1))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['binary_accuracy'])
# Confusion Matrix에서 어떤 값을 취할 것인지
# Accuracy, Precision, Sensitivity, Specificity
# 근데 여기서는 적용이 안된다고 함. 왜?
# 사실 여기서는 Precision이 중요


fit_hist = model.fit(x_train, y_train, batch_size=50,epochs=500, validation_split= 0.2, verbose = 1)
# 20% 복원추출 랜덤하게

plt.plot(fit_hist.history['binary_accuracy'])
plt.plot(fit_hist.history['val_binary_accuracy'])
plt.show()

score = model.evaluate(x_test,y_test,verbose=0)
print('Keras DNN model loss :',score[0])
print('Keras DNN model accuracy :',score[1])

# Evaluation - 학습은 하지않고 검증만 한다.