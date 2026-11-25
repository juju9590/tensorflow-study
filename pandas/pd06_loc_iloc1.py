import pandas as pd
print(pd.__version__) #3.0.6

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
]

# 데이터는 리스트 상태  >> 판다스로 바꾸겠음

index = ['031','059', '033', '045', '023']
columns =['종목명','시가','종가']

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)

#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500

# print(df[0]  # raise KeyError(key) from err
# print(df['031']) # raise KeyError(key) from err
# print(df['종목명']) #암기 *** 판다스 열행 >>>> 열이 우선이다 
# ㄴ 보통은 행열 기준이지만, 판다스는 열행으로 순서가 바뀐다.. 암기 필수
# print("======= 아모래 시가 출력 =======")

# print(df[3,1])  #raise KeyError(key) from err
# print(df[3][1])  #raise KeyError(key) from err

# print(df['045','종목명']) #raise KeyError(key) from err
# print(df['045']['종목명']) #raise KeyError(key) from err

# print(df['종목명'] ['045']) #아모레 *** 판다스 열행 **** 컬럼이 기준

##########################################################
# loc : 인덱스 기준으로 행 데이터 추출
# iloc : 행 번호를 기준으로  행 데이터 추출
    #  ************int loc 로 외워라 !!! **********
##########################################################
# print("======= 아모래 뽑기 =======")
# print(df.iloc[3])
# ======= 아모래 뽑기 =======
# 종목명     아모레
# 시가     3500
# 종가     6000
# Name: 045, dtype: str

# print(df.iloc['045']) #Cannot index by location index with a non-integer key
# 정수(Integer)가 아닌 형태(문자열, UUID, 실수, 객체 등)로 된 키(Key)

# print(df.loc[3]) #raise KeyError(key) from err/KeyError: 3

# print(df.loc['045'])
# ======= 아모래 뽑기 =======
# 종목명     아모레
# 시가     3500
# 종가     6000
# Name: 045, dtype: str

# print("======= 네이버 뽑기 =======")
# print(df.iloc[4])

# ======= 네이버 뽑기 =======
# 종목명     네이버
# 시가      100
# 종가     1500
# Name: 023, dtype: str

# print(df.loc['023'])
# 종목명     네이버
# 시가      100
# 종가     1500
# Name: 023, dtype: str

# print("======= 아모레 종가 =======")
# print(df.iloc[3]['종가'])
# ======= 아모레 종가 =======
# 6000
# print(df.iloc[3][2]) #raise KeyError(key) from err
# print(df.iloc[3, 2]) #6000
# print(df.loc['045','종가']) #6000
# print(df.iloc[3,'종가']) #ValueError

# print(df.loc['045','종목명']) #아모레/
# print(df.loc['045'][2]) #raise KeyError(key) from err
# print(df.loc['045',2]) #raise KeyError(key) from err

print("======================================")

print(df.iloc[3].iloc[2]) #6000
print(df.iloc[3].loc['종가']) #6000

print(df.loc['045'].loc['종가']) #6000
print(df.loc['045'].iloc[2]) #6000

#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# 023  네이버   100  1500
# ======================================
# 6000
# 6000
# 6000
# 6000














