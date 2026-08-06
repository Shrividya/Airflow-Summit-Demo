class _Metric:
    def __init__(self, name):
        self.name = name


faithfulness = _Metric("faithfulness")
context_precision = _Metric("context_precision")
context_recall = _Metric("context_recall")
