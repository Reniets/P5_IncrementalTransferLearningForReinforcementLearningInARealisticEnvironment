import copy


class UncertaintyLog:
    def __init__(self, seen_observations):
        self.observations = copy.deepcopy(seen_observations)

    def len(self):
        return len(self.observations)

    def getCount(self, index):
        if index >= len(self.observations):
            return 0
        else:
            return self.observations[index][0]  # [0] is the counter [1] is the image
