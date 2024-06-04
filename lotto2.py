import pandas as pd
import numpy as np
import random
from collections import Counter

def select_numbers(counts):
    sorted_counts = counts.sort_values(ascending=False)
    ranks = sorted_counts.unique()
    
    selected_numbers = []
    
    # 조건 1과 2
    if len(sorted_counts[sorted_counts == ranks[0]]) <= 2:
        selected_numbers.extend(sorted_counts[sorted_counts == ranks[0]].index.tolist())
        if len(ranks) > 1:
            selected_numbers.extend(sorted_counts[sorted_counts == ranks[1]].index.tolist())
        if len(ranks) > 2:
            selected_numbers.extend(sorted_counts[sorted_counts == ranks[2]].index.tolist())
    else:
        selected_numbers.extend(sorted_counts[sorted_counts == ranks[0]].index.tolist())
    
    # 조건 3
    if len(ranks) > 1 and len(sorted_counts[sorted_counts == ranks[1]]) >= 2:
        selected_numbers.extend(sorted_counts[sorted_counts == ranks[1]].index.tolist())
    
    # 조건 4는 이미 조건 1과 2에서 3순위를 포함시키는 경우만 처리하므로 별도의 처리가 필요 없습니다.
    
    return selected_numbers

def position(df): 
    recent = df.iloc[-24:]
    result = []
    for i in range(6):
        counts = recent[i].value_counts()
        selected_numbers = (select_numbers(counts))
        #print("선택된 숫자들:", selected_numbers)
        result.extend(selected_numbers)
    #result = set(result)
    # print(f"position -- > {list(set(all_number) - set(result))}")
    return list(set(all_number) - set(result))   #제외수 리스트


def history_win(df, input_numbers):
    # 등수별로 몇 회 당첨되었는지 저장할 변수를 초기화합니다.
    rank_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, '꽝': 0}

    # 각 행(회차)별로 당첨 번호와 비교합니다.
    for index, row in df.iterrows():
        winning_numbers = set(row[:6])
        bonus_number = row[6]
        
        # 당첨 번호와 입력 번호의 교집합을 구합니다.
        matched_count = len(winning_numbers.intersection(input_numbers))
        
        # 등수를 판단합니다.
        if matched_count == 6:
            rank_counts[1] += 1
        elif matched_count == 5 and bonus_number in input_numbers:
            rank_counts[2] += 1
        elif matched_count == 5:
            rank_counts[3] += 1
        elif matched_count == 4:
            rank_counts[4] += 1
        elif matched_count == 3:
            rank_counts[5] += 1
        else:
            rank_counts['꽝'] += 1

        if rank_counts[1] + rank_counts[2] + rank_counts[3] == 0:
            return True
    return False
    
#하나의 번호대를 제거 시키고 싶어.
def exclude_range(df, aa = -1):
    def get_range(number):
        return number // 10

    def find_missing_ranges(draw_numbers):
        appeared_ranges = set(get_range(num) for num in draw_numbers)
        missing_ranges = set(range(5)) - appeared_ranges
        return list(missing_ranges)

    def result(df):
        all_results = []
        for i, row in df.iterrows():
            draw_numbers = row.values
            missing_ranges = find_missing_ranges(draw_numbers)
            all_results.extend(missing_ranges)    
        return all_results
    recent_10_results = result(df.iloc[aa-12:aa]) #최근 10개
    counter = Counter(recent_10_results)
    # for number_range in range(5):
    #     print(f"{number_range*10}번대: {counter[number_range]}회")
    
    # value가 가장 작은 key를 찾기
    min_count = min(counter.values())
    min_keys = [k for k, v in counter.items() if v == min_count]

    # 최소 값 key 중 첫 번째를 선택
    min_key = min_keys[0]
    
    return min_key * 10


def main():
    df = pd.read_csv('lotto.csv', header=None)
    last_win = [3,6,21,30,34,35,22]
    all_number = np.arange(1, 46)   #1부터 45까지의 숫자를 담은 어레이

    
    # 역대 평균값들. 여기서 2표준편차 이내에 들어오면 합격으로 함.
    mean = np.array(df.mean(axis=1))
    _min = mean.mean() - mean.std()
    _max = mean.mean() + mean.std()


    # 지난 24회차 동안 6회 이상 출현시 제외수로 ㄱㄱ
    ccc = pd.Series(df.iloc[-24:].values.flatten()).value_counts()
    exclude_numbers = np.array([int(item) for item in list(ccc[ccc >= 6].index)])

    #최근 2회 연속으로 나온 숫자는 제외시킴
    recent_draws = df.tail(2)
    repeated_numbers = set(recent_draws.iloc[0]).intersection(set(recent_draws.iloc[1]))
    exclude_numbers = np.append(exclude_numbers, list(repeated_numbers))

    #지난 13회차 동안 0회 숫자는 고정수로(고정수는 무조건 뽑는건 아니고 뽑을 예정 수에 들어감)
    cc = pd.Series(df.iloc[-13:].values.flatten()).value_counts()
    num = np.delete(all_number, cc.index-1)

    ex_range = exclude_range(df)
    exclude_numbers = np.append(exclude_numbers, np.arange(ex_range, ex_range+10))

    carried_number = np.array(df.iloc[-1][:7],dtype=int)      # 이월수
    carried_number = np.setdiff1d(carried_number, exclude_numbers)
    reverse_number = np.array(46 - df.iloc[-1][:7],dtype=int) # 역수
    reverse_number = np.setdiff1d(reverse_number, exclude_numbers)

    n = 0
    while n < 10:
        select_carried = carried_number[random.randrange(0, len(carried_number))] #이월수 1개 랜덤선택
        select_reverse = reverse_number[random.randrange(0, len(reverse_number))] #역수 1개 랜덤선택
        
        exclude_numbers = np.concatenate((exclude_numbers, carried_number, reverse_number))
        exclude_numbers = np.concatenate((position(df), exclude_numbers))

        numbers = np.setdiff1d(all_number, exclude_numbers)
        
        selected = np.random.choice(numbers, size=4, replace=False)
        selected = np.append(selected, [select_carried, select_reverse])
        
        if history_win(df, selected) and _min < np.mean(selected) < _max:
            selected = np.sort(selected)
            last_digit = [str(x)[-1] for x in selected] #끝수 규칙 적용(끝수가 2개 같은애가 존재하도록)
            if len(set(last_digit)) == 5:
                intersection_count = len(set(selected).intersection(last_win[:6]))

                if intersection_count == 3:
                    print(f"{selected} --- 5등 당첨")
                elif intersection_count == 4:
                    print(f"{selected} --- 4등 당첨")
                elif intersection_count == 5:
                    if last_win[-1] in selected:
                        print(f"{selected} --- 2등 당첨")
                    else:
                        print(f"{selected} --- 3등 당첨")
                elif intersection_count == 6:
                    print(f"{selected} --- 1등 당첨")
                else:
                    print(f"{selected}")
                n += 1
            

if __name__ == '__main__':
    df = pd.read_csv('lotto.csv', header=None)
    last_win = [3,6,21,30,34,35,22]
    all_number = np.arange(1, 46)   #1부터 45까지의 숫자를 담은 어레이
    main()
