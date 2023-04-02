

class timeMetrics:
    def __init__(self):
        self.count = 0
        self.time = 0
        self.avg = 0
        self.min = 999
        self.max = 0

    def update(self, exe_time):
        self.count += 1
        self.time += exe_time
        self.avg = self.time / self.count
        self.min = min(self.min, exe_time)
        self.max = max(self.max, exe_time)
        

    def get_results(self):
        return {'avg': self.avg, 'min': self.min, 'max': self.max}
    
    def to_str(self, metrics):
        string = "\n"
        for k, v in metrics.items():
            string += "%s: %f\n"%(k, v)
        return string