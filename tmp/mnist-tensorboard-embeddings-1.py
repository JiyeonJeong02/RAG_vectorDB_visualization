# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorboard.plugins import projector
import numpy as np

# 현재 작업 디렉토리 경로
PATH = os.getcwd()

# 로그 디렉토리 설정
LOG_DIR = PATH + '/mnist-tensorboard/log-3'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

# 디렉토리가 없으면 생성
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Eager Execution 비활성화
tf.compat.v1.disable_eager_execution()

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 784)  # 이미지를 (784,) 형태로 변환
images = tf.Variable(x_test, name='images')

# 메타데이터 파일 저장
with open(metadata, 'w') as metadata_file:
    for row in range(10000):
        c = y_test[row]  # 레이블을 그대로 사용합니다.
        metadata_file.write('{}\n'.format(c))



# TensorFlow 세션 대신 TensorFlow 2.x의 `tf.compat.v1.Session` 사용
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(tf.compat.v1.summary.FileWriter(LOG_DIR), config)
    
# TensorBoard를 호출하려면 해당 디렉토리로 이동한 후 전체 경로를 호출해야 합니다.
# tensorboard --logdir=/Technical_works/tensorflow/mnist-tensorboard/log-3 --port=6006
# 캐시 문제로 안될 수도 있으니 캐시를 초기화 하는 명령어가 필요하면 아래를 cmd에서 실행할 것(브라우저 캐시도 영향을 끼칠 수 있으니, 다른 브라우저에서 시도)
# tensorboard --logdir=C:\coding\tensorflow\mnist-tensorboard\log-3 --port=6006 --reload_multifile=True
