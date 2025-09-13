import torch


# RandomSampler: train 데이터를 랜덤하게 뽑아주는 역할
class RandomSampler:
    def __init__(self, full_size, batch_size, last_drop=True):
        self.full_size = full_size  # 전체 데이터 셋 사이즈
        self.batch_size = batch_size  # 미니 배치 사이즈
        self.last_drop = last_drop
        self.shuffle()  # 랜덤 index 벡터 생성

    def shuffle(self):
        self.index = torch.randperm(self.full_size)
        self.start = 0  # index의 시작점
        self.end = self.batch_size  # index의 끝점

    def get_random_idx(self):  # 랜덤한 index를 return 해주는 함수
        if self.end > len(self.index):  # end가 index를 넘어간 경우 두가지 way가 있음
            if self.last_drop:  # last_drop가 True이면 shuffling
                self.shuffle()
            elif self.start >= len(
                self.index
            ):  # start가 index를 넘어간 경우는 shuffling
                self.shuffle()

        idx = self.index[self.start : self.end]  # index 뽑아주기
        self.start += self.batch_size  # 미리 다음 mini-batch로 index를 옮김
        self.end += self.batch_size
        return idx
