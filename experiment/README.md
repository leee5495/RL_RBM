# Enhanced Collaborative Filtering with Reinforcement Learning 실험 결과

### 실험 데이터 설명
MovieLens 평가 데이터세트를 이용해 추천 시스템에 대한 성능 평가 진행
- MovieLens 웹사이트에서 수집된 영화 평가 데이터 세트
- 추천 시스템의 성능을 검정하기 위한 벤치마크 데이터 세트로 자주 사용
- 100,000개의 평가 데이터로 구성된 MovieLens100K와 1,000,000개의 평가 데이터로 구성된 MovieLens1M을 사용함

##
### 기본 RBM과의 성능 비교 (compare_rbm.py)
**MovieLens100K 결과**
![image](https://user-images.githubusercontent.com/39192405/93019674-7bcc0080-f613-11ea-8844-c96b4651236a.png)

    | RLRBM | RBM
  -- | -- | --
  HR@10 | 0.1481 | 0.1217
  HR@25 | 0.2169 | 0.2169
  ARHR | 0.05986 | 0.05152


**MovieLens1M 결과**
![image](https://user-images.githubusercontent.com/39192405/93019679-7ff81e00-f613-11ea-8fe0-c23c9138c6dc.png)

    | RLRBM | RBM
  -- | -- | --
  HR@10 | 0.09272 | 0.08940
  HR@25 | 0.1813 | 0.1763
  ARHR | 0.04423 | 0.04353


- 아이템과 사용자 수가 더 적고 평가의 density가 높은 MovieLens100K에 대해선 눈에 띄는 성능 향상을 볼 수 있다.
- 하지만 아이템과 사용자 수가 많고 density가 낮은 MovieLens1M에서는 큰 변화를 볼 수 없다.

  - MovieLens1M의 결과는 여러 반복 실험을 걸쳐 얻은 가장 좋은 결과만 사용

##
### 지도학습을 통해 Left out item에 대한 추가적인 학습이 진행된 RBM과 성능 비교 (compare_supervised.py)

**MovieLens100K 결과**
![image](https://user-images.githubusercontent.com/39192405/93019768-0ad91880-f614-11ea-9cf2-6cfcbfb58b5f.png)

    | RLRBM | SUPERVISED
  -- | -- | --
  HR@10 | 0.1481 | 0.1217
  HR@25 | 0.2169 | 0.2275
  ARHR | 0.05986 | 0.05036

**MovieLens1M 결과**
![image](https://user-images.githubusercontent.com/39192405/93019797-32c87c00-f614-11ea-8296-b15c7ec2c950.png)

    | RLRBM | SUPERVISED
  -- | -- | --
  HR@10 | 0.09271 | 0.08858
  HR@25 | 0.1813 | 0.1738
  ARHR | 0.04423 | 0.04557

- 같은 Left out item에 대한 학습을 강화학습으로 했을 때와 지도학습으로 했을 때의 성능 비교 결과 강화학습의 성능이 대부분 더 높은 것을 확인했다.

- 지도학습은 하나의 사용자 상태에 대해 하나의 분명한 정답을 학습시키는 것에 반면, 강화학습은 사용자의 상태에 대해 확률적으로 보상을 분배하기 때문에 더 효과적인 것으로 보인다.

##
### 강화학습 훈련을 진행할 때 선택하는 아이템의 개수 K에 따른 효과 비교 (compare_k.py)

![image](https://user-images.githubusercontent.com/39192405/93019820-699e9200-f614-11ea-8670-5834469d5c45.png)

- 작은 크기의 데이터세트에서는 작은 K 값의 성능이 대부분 더 높은 경향을 띄는 것을 볼 수 있다.

- 큰 크기의 데이터세트에서는 상대적으로 더 큰 K 값을 사용할 때 성능이 높아지는 것을 확인할 수 있었다.

- 아이템과 사용자의 수가 많은 데이터일수록 더 큰 K 값을 사용해 사용자의 피드백을 더 많이 얻는 것이 강화학습 훈련에 더 효과적인 것으로 보인다.
