#### example of performance functions

class PerformanceFunction(object):
    pass

class SmoothDelta(PerformanceFunction):
    @staticmethod
    def performance(prediction, output):
        return 0.5 * (prediction - output)**2

    @staticmethod
    def dperformance(prediction, output):
        return output - prediction
