# Train logger
class TrainLogger:
    def __init__(self):
        pass

# Tensorbord logger
class TensorBoardLogger(TrainLogger):
    pass


# Result object
class Result(object):
    def __init__(
            self, main_score: float, log_header: str, log_line: str, detailed_results: str, metrics: dict =None
    ):
        self.main_score: float = main_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.detailed_results: str = detailed_results
        self.metrics: dict = metrics

