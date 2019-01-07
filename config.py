#-*- coding:utf-8 -*-

# 불변값
big = 57
medium = 552
small = 3190
detail = 404
img_feature_length = 2048

# 데이터 경로 정의
train_db_dir = '../'  # 학습 데이터의 경로
test_db_dir = '../'  # dev 데이터의 경로
result_model_dir = '../'  # 학습된 모델이 저장될 경로

# 학습 파라미터
embedding = 32
strmaxlen = 90
epochs = 100

learning_rate = 0.001
character_size = 251
batch_size = 256 #1024

input_size = embedding * strmaxlen

