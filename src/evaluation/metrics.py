def summarize_scores(scores):
    return {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
    }
