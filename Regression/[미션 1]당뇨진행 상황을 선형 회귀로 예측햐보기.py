import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""
    1. 학습을 위한 준비가 완료된 데이터를 반환하는
       load_data() 함수를 구현합니다.

       Step01. 당뇨병 관련 데이터셋을 (X, y)의 형태로 불러옵니다. 

       Step02. 모델 학습을 위해 데이터를 
               학습용(80%)/테스트용(20%)로 분리합니다.
               (random_state = 100)
"""


def load_data():
    X, y = load_diabetes(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=100)
    return train_X, test_X, train_y, test_y


"""
    2. 회귀 모델을 불러오고,
       테스트 데이터에 대한 예측 결과를 반환하는
       reg_model() 함수를 구현합니다.
"""


def reg_model(train_X, test_X, train_y):
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    pred = lr.predict(test_X)
    return pred


"""
    3. 구현한 회귀 모델의 r_square 값을
       반환하는 r_square() 함수를 구현합니다.

"""


def r_square(pred, test_y):
    r2 = r2_score(test_y, pred)

    return r2


"""
    4. 구현한 함수들을 활용하여 
       당뇨병 데이터에 대한 회귀를 진행하는 
       main() 함수를 구현합니다.
"""


def main():
    train_X, test_X, train_y, test_y = load_data()
    r2 = r_square(test_y, reg_model(train_X, test_X, train_y))

    print("r2 score : ", r2)


if __name__ == "__main__":
    main()