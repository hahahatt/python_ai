import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from tqdm import tqdm


class KMeansClustering:
    """
    고객 데이터를 KMeans 클러스터링을 통해 분석하는 클래스

    @Attributes
        __init__(self,file_path): 클래스 초기화, 데이터 로드 및 변수 초기화
    
    @Methods
        SetData():        # 스케일링 및 학습, 테스트 데이터 설정
        FindOptimalK():   # 엘보우 그래프 생성 및 최적 k 결정
        DoKmeans():       # KMeans 수행
        EvaluateKmeans(): # KMeans 실루엣 점수 확인
        ShowKmeans():     # 그래프 출력
        AnalyzeResult():  # 결과 분석
    """
    def __init__(self, file_path):
        """
        변수 초기화 및 파일 로드

        @param self
        @param file_path : 파일 경로
        @return : 없음
        """
        print('')
        print('='*20, ' 1. 파일로드 및 초기화 ', '='*20)

        self.df = pd.read_csv(file_path)    # 파일 로드
        self.scaler = StandardScaler()      # 객체 생성

        self.kmeans = None                  # 변수 선언
        self.optimal_k = 0                  # 변수 선언
        self.X_train = None                 # 변수 선언
        self.X_test = None                  # 변수 선언
        self.train_labels = None            # 변수 선언
        self.test_labels = None             # 변수 선언
        self.cluster_centers = None         # 변수 선언
        self.score = 0

    def SetData(self):
        """
        필요한 열 추출, 스케일링 및 학습 데이터  생성

        @param: self
        @return : 없음
        """
        print('')
        print('='*20, ' 2. 전처리 ', '='*20)

        # 필요한 열 추출
        data = self.df[['Annual Income (k$)', 'Spending Score (1-100)']]
        print("결측치:")
        print(data.isnull().sum()) # 결측치 확인

        scaled = self.scaler.fit_transform(data) # 스케일링 수행`1`
        self.X_train, self.X_test = train_test_split(scaled, test_size=0.2, random_state=42) # 학습 및 테스트 데이터 분리

    def FindOptimalK(self):
        """
        최적의 클러스터 수를 찾기 위한 엘보우 방법 적용

        @param self
        @return : 없음
        """
        print('')
        print('='*20, ' 3. 최적 K 찾기 및 엘보우 그래프 출력 ', '='*20)

        plt.rc('font', family='Malgun Gothic')

        li = []  # 리스트 선언
        ran = range(1,11)   # 범위값 지정
        for i in tqdm(ran, desc="Elbow 진행 중"): # tqdm으로 진행률 확인
            model = KMeans(n_clusters=i, random_state=42)   # 클러스터 별 객체 생성
            model.fit(self.X_train)         # 모델 적용
            li.append(model.inertia_)  # 모델의 inertia 값 리스트 저장
        
        # 엘보우 시각화
        plt.figure(figsize=(8, 5))
        plt.plot(ran, li, marker='o')
        plt.title('엘보우 그래프')
        plt.xlabel('클러스터 수')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()


        # 기울기 계산
        for i in range(1,10):
            print(f'리스트[{i-1}] - 리스트[{i}] 차이(기울기) : ',li[i-1] - li[i])

        print("기울기가 급격히 완화되는 지점(엘보우 포인트), 적절한 K 값은 5이다.")
        self.optimal_k = 5


    def DoKmeans(self):
        """
        주어진 클러스터 수로 KMeans 모델 학습 및 예측 진행

        @param self
        @return : 없음
        """
        print('')
        print('='*20, ' 4. KMeans 적용 ', '='*20)

        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42) # KMeans 객체 생성
        self.train_labels = self.kmeans.fit_predict(self.X_train)        # 훈련 데이터 클러스터링 및 반환
        self.test_labels = self.kmeans.predict(self.X_test)              # X_test 예측
        self.cluster_centers = self.kmeans.cluster_centers_

    def EvaluateKmeans(self):
        """
        테스트 데이터에 대한 실루엣 점수를 계산하여 클러스터링 품질을 평가

        @param self
        @return : 없음
        """

        print('')
        print('='*20, ' 5. 실루엣 점수 출력 ', '='*20)

        self.score = silhouette_score(self.X_test, self.test_labels) # 클러스터링 품질 평가
        print(f"Silhouette Score: {self.score:.4f}") # 실루엣 점수 출력
        print('실루엣 점수가 0과 1 중 그나마 1에 가까움')
        print('=> 데이터가 그나마 자신의 클러스터에 속하고, 다른 클러스터와 분리됨')


    def ShowKmeans(self):
        """
        학습 및 테스트 데이터를 시각화

        @param self
        @return : 없음
        """
        print('')
        print('='*20, ' 6. 산점도 그래프 출력 ', '='*20)

        plt.rcParams['font.family'] = "Malgun Gothic"   # 한글 폰트 설정
        plt.rcParams['axes.unicode_minus'] = False      # 수학 기호 폰트 유지

        plt.figure(figsize=(12, 5))

        # 학습 데이터
        plt.subplot(1, 2, 1)    # 서브플롯 생성

        # scatterplot 생성
        # 훈련 데이터 기준
        custom_palette = ['red', 'blue', 'green', 'orange', 'purple']
        sns.scatterplot(x=self.X_train[:, 0], y=self.X_train[:, 1], hue=self.train_labels, palette=custom_palette)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1],
                    s=200, c='black', marker='X', label='Centroids')
        
        plt.title('훈련 데이터 클러스터링')
        plt.xlabel('연 수입')
        plt.ylabel('소비 점수')
        plt.legend()

        # scatterplot 생성
        # 테스트 데이터 기준
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=self.X_test[:, 0], y=self.X_test[:, 1], hue=self.test_labels, palette=custom_palette)
        plt.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1],
                    s=200, c='black', marker='X', label='Centroids',)
        plt.title('테스트 데이터 클러스터링')
        plt.xlabel('연 수입')
        plt.ylabel('소비 점수')

        plt.tight_layout()
        plt.legend()

        plt.show()

    def AnalyzeResult(self):
        """
        클러스터별 center에 해당하는 연 수입과 소비 점수 출력 및 해석

        @param self
        @return 없음
        """
        print('')
        print('='*20, ' 7. 연 수입 및 소비점수 결과 ', '='*20)
        for i, center in enumerate(self.cluster_centers):
            # 연수입, 소비점수 출력
            print(f"Cluster {i}: Annual Income(연 수입) = {center[0]:.2f}, Spending Score(소비 점수) = {center[1]:.2f}")

        print('')
        print('Cluster 0 : 적당한 연 수입(적당), 적당한 소비 점수(적당) ==> 일반 고객')
        print('Cluster 1 : 낮은 연 수입(저소득), 낮은 소비 점수(저소비) ==> 관심 고객 X')
        print('Cluster 2 : 낮은 연 수입(저소득), 높은 소비 점수(고소비) ==> 충성? 좋은? 고객')
        print('Cluster 3 : 높은 연 수입(고소득), 높은 소비 점수(고소비) ==> VIP 고객')
        print('Cluster 4 : 높은 연 수입(고소득), 낮은 소비 점수(저소비) ==> 적극적 마케팅 필요한 고객')
        


if __name__ == "__main__":
    """
    메인 함수
    """
    cluster = KMeansClustering('./aaa/mall_customers.csv') # 객체 생성
    cluster.SetData()        # 스케일링 및 학습, 테스트 데이터 설정
    cluster.FindOptimalK()   # 엘보우 그래프 생성 및 최적 k 결정
    cluster.DoKmeans()       # KMeans 수행
    cluster.EvaluateKmeans() # KMeans 실루엣 점수 확인
    cluster.ShowKmeans()     # 그래프 출력
    cluster.AnalyzeResult()  # 결과 분석