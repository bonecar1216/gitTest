# AI 개발 실무 2차 레포트
# 미래학부_이창훈_202232421
# 레벤슈타인 거리를 이용한 챗봇 구현하기

# 챗봇 구현
# 1) 질문&답변 데이터(ChatbotData.csv)에서 질문과 답변 데이터 로드
# 2) 질문과 답변을 각각 추출하여 question과 answer 리스트에 저장
# 3) 사용자 질문 입력
# 4) 챗봇이 가지고 있는 qustion 리스트 내 질문들과 사용자 질문과의 레벤슈타인 거리 계산
# 5) 레벤슈타인 거리 계산 값을 리스트에 저장
# 6) 모든 거리 계산이 끝난 후, 리스트 내 가장 작은 값의 인덱스를 확인
# 7) answer 리스트에서 해당 인덱스의 답변 값을 사용자 질문에 대한 답변으로 채택하여 출력

# 필요한 라이브러리 불러오기

import pandas as pd                         # 질문&답변 데이터(ChatbotData.csv)를 로드하기 위해 pandas 라이브러리를 불러옴
import numpy as np

# <챗봇 구현>

# 챗봇 클래스 선언
class ChatBot_LD:

    # 인스턴스 생성
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)

    # 질문&답변 데이터를 로드하여 리스트에 저장
    def load_data(self, filepath):
        data = pd.read_csv(filepath)        # 질문&답변 데이터 csv 파일을 불러들여 pandas 데이터프레임으로 변환하여 data 변수에 저장
        questions = data['Q'].tolist()      # 데이터프레임에서 질문 열만 뽑아 questions 리스트에 저장
        answers = data['A'].tolist()        # 데이터프레임에서 답변 열만 뽑아 answers 리스트에 저장
        return questions, answers           # questions, answers 반환
    

    # 사용자 질문과 questions 리스트 내 질문과의 레벤스타인 거리 계산
    def calc_distance(self, input_sentence):
        LD_results = []                             # 사용자 질문과 qustions 질문 리스트 간의 레벤슈타인 거리를 저장하기 위한 리스트 선언
        
        # questions 리스트 내 질문을 하나씩 꺼내어 사용자의 질문과의 레벤슈타인 거리를 계산
        for q in self.questions:
            if input_sentence == q: 
                LD_results.append(0)                # 사용자 질문과 질문 데이터(q)가 같으면 0을 결과 리스트에 추가
                return LD_results                   # 결과 리스트를 반환하면서 조기 종료

            i_len = len(input_sentence)             # 사용자 질문의 길이
            q_len = len(q)                          # 질문 데이터의 길이 

            # 사용자 질문과 question의 질문의 문장 길이에 맞는 메트릭스 생성
            matrix = [[] for i in range(i_len+1)]        # 리스트 컴프리헨션을 사용하여 1차원 초기화
            for i in range(i_len+1):                     
                matrix[i] = [0 for j in range(q_len+1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
            
            # 각 인덱스 값을 각 위치의 초깃값으로 저장
            for i in range(i_len+1):
                matrix[i][0] = i
            for j in range(q_len+1):
                matrix[0][j] = j
            
            # 레벤슈타인 거리 계산 구문   
            for i in range(1, i_len+1):             
                ic = input_sentence[i-1]            # 사용자 질문에서 한 글자씩 꺼내어 저장 -> ic
        
                for j in range(1, q_len+1):
                    qc = q[j-1]                     # questions 질문에서 한 글자씩 꺼내어 저장 -> qc

                    # ic와 qc 비교
                    cost = 0 if (ic == qc) else 1   # ic와 qc가 같은 경우 cost를 0, 동일하지 않으면 1로 저장
                    
                    # 문자 제거, 삽입, 변경 중 가장 작은 값을 해당 인덱스의 매트릭스에 저장
                    matrix[i][j] = min([            
                        matrix[i-1][j] + 1,         # 문자 제거인 경우 위쪽에서 +1
                        matrix[i][j-1] + 1,         # 문자 삽입인 경우 왼쪽 수에서 +1   
                        matrix[i-1][j-1] + cost     # 문자 변경인 경우 대각선에서 +1 (cost = 1), 문자가 동일하면 대각선 숫자 복사
                    ])
            
            LD_results.append(matrix[i_len][q_len]) # 레벤슈타인 거리 값을 리스트에 저장
        
        return LD_results

    def find_best_answer(self, input_sentence):
        
        # 사용자 질문이 입력되지 않을 경우, 질문 입력 요청 메세지 출력
        if len(input_sentence) == 0:
            return '질문이 입력되지 않았습니다. 질문을 입력해주세요.'
        
        else: 
            LD_results = self.calc_distance(input_sentence)     # calc_distance 함수를 통해 레벤슈타인 거리를 계산한 결과 리스트를 받음
            best_match_index = np.argmin(LD_results)            # LD_results 리스트에서 가장 작은 값을 갖는 값의 인덱스를 찾아서 저장
                                                                # : 레벤슈타인 거리 값이 작아야 유사한 문장
            return self.answers[best_match_index]               # 사용자 질문과 유사한 질문에 대한 답변을 출력 
                                                                # : 유사한 질문과 답변은 동일 인덱스에 위치함



# <챗봇 실행>

# 질문&답변 데이터 csv 파일 경로 지정
filepath = 'ChatbotData.csv'

# 레벤슈타인 거리를 이용한 챗봇 인스턴스 생성
chatbot = ChatBot_LD(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복
while True:
    input_sentence = input('You: ')                         # 사용자의 질문을 입력받아 input_sentence 변수에 저장
    if input_sentence.lower() == '종료':                     # 반복문 종료 조건
        break
    response = chatbot.find_best_answer(input_sentence)     # input_sentence를 find_best_answer함수의 입력값으로 넣어서 나온 적절한 답변은 response 변수에 저장
    print('Chatbot:', response)                             # 사용자 질문에 대한 답변 출력