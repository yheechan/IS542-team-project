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