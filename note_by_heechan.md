# note by heechan


## 241101 9주차 금요일
* RICC
    * 이는 train 노드들을 랜덤하게 뽑아 CC 알고리즘을 수행한다.
    * 위 과정을 다양한 랜덤 train node들을 가지고 반복한다.
* dataset에 ``train*`` 파일들은 무엇인가?
    * negative (benign)과 positive (sybil) node들이 있다.
    * facebook 경우 각 benign과 sybil node 100개 씩
* dataset에 ``target*`` 파일들은 무엇인가?
    * sybilSCAR들이 맞춰야하는 sybil node들이다.
    * 어떤 기준인지는 모르겠으나, 여기 포함된 node들도 ``train*`` 파일에 포함 된 node들도 있다.

* check_FN_nodes() need to more analysis time.


## 241121 12주차 목요일
### 질문 리스트
1. ``check_FN_nodes()``함수에서 ``target.txt`` 기반으로 공격 그래프에 대한 sybil classification을 평가한다. 이때 ``target.txt`` **전체 그래프**라는 파일로 부터 랜덤 ``N``개의 sybil node와 랜덤 ``N``개의 benign node를 선택하여 평가 지표로 사용된다. 근데 **전체 그래프**, 이 파일은 대체 어디있는가? ``initial_files/<subject>/prior_0(0.5).txt`` 파일을 보면 -0.5는 benign, 0.5는 sybil이라고 하는데, 통계 결과는 다음과 같이 나온다.
    ```
    # initial_files/Facebook/prior_0(0.5).txt
    Total number of scores: 8078
    Total nodes: 8078
    Total zero nodes: 7878
    Total benign nodes: 100
    Total sybil nodes: 100

    # initial_files/Enron/prior_0(0.5).txt
    Total number of scores: 67392
    Total nodes: 67392
    Total zero nodes: 67192
    Total benign nodes: 100
    Total sybil nodes: 100
    ```

## 241127 13주차 수요일
* 첫 4039 node는 항상 benign으로 둬야할 것. 즉, 이 node들은 remove 할 수 없게 구현해야할 것.
    * 이유: 실제 공격 시나리오 생각해보면, 공격자는 자기네 계정을 추가 혹은 삭제 할 수 있지, benign계정은 건드릴 수 없다.
