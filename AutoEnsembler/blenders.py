class BlenderBase:
    def __init__(self, name):
        self.name = name

    def blend(self, scores, *args):
        raise NotImplementedError

        
class PowerBlend(BlenderBase):
    def __init__(self):
        super().__init__("")

    def blend(self, scores, lin_coefs, exp_coefs):
        result = 0
        for index, score in enumerate(scores):
            result += lin_coefs[index] * np.power(score, exp_coefs[index])
        return result
