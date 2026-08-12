"""Fake `ragas.evaluate`: approximates with word-overlap heuristics instead
of a real LLM judge, just enough to exercise the quality-gate branching
logic in include/evaluate.py offline. Scores aren't meaningful as a real
quality signal."""


def _words(text: str) -> set:
    return set(w.strip(".,!?()[]").lower() for w in text.split() if len(w) > 3)


def evaluate(dataset, metrics, llm=None):
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]

    faithfulness_scores = []
    precision_scores = []
    recall_scores = []

    for answer, contexts, gt in zip(answers, contexts_list, ground_truths):
        context_words = set()
        for c in contexts:
            context_words |= _words(c)
        gt_words = _words(gt)

        overlap_with_context = len(gt_words & context_words) / max(len(gt_words), 1)
        faithfulness_scores.append(min(1.0, 0.5 + 0.5 * overlap_with_context))
        precision_scores.append(min(1.0, 0.4 + 0.6 * overlap_with_context))
        recall_scores.append(min(1.0, 0.4 + 0.6 * overlap_with_context))

    n = len(questions)
    result = {}
    metric_names = {m.name for m in metrics}
    if "faithfulness" in metric_names:
        result["faithfulness"] = sum(faithfulness_scores) / n
    if "context_precision" in metric_names:
        result["context_precision"] = sum(precision_scores) / n
    if "context_recall" in metric_names:
        result["context_recall"] = sum(recall_scores) / n
    return result
