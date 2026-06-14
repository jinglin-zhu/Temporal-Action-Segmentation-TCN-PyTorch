# some parts of this code are a modified version from https://github.com/yabufarha/ms-tcn
from typing import List, Union, Iterable
import numpy as np


def careful_divide(correct: int, total: int, zero_value: float = 0.0) -> float:
    if total == 0:
        return zero_value
    else:
        return correct / total

def get_labels_start_end_time(frame_wise_labels, bg_class=[]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=[]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


class Edit(object):
    def __init__(self, ignore_ids: List[int] = []):
        self.ignore_ids = ignore_ids
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.values = []

    def add(
        self, targets: List[int], predictions: List[int]
    ) -> float:
        current_score = edit_score(
            recognized=predictions,
            ground_truth=targets,
            bg_class=self.ignore_ids,
        )

        self.values.append(current_score)
        return current_score

    def summary(self) -> float:
        if len(self.values) > 0:
            return np.array(self.values).mean()
        else:
            return 0.0


class MoFAccuracyMetric(object):
    def __init__(self, ignore_ids: List[int] = []):
        self.ignore_ids = ignore_ids
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.total = 0
        self.correct = 0

    def add(self, targets: List[int], predictions: List[int]) -> float:
        assert len(targets) == len(predictions)
        targets, predictions = np.array(targets), np.array(predictions)

        mask = np.logical_not(np.isin(targets, self.ignore_ids))
        targets, predictions = targets[mask], predictions[mask]

        current_total = len(targets)
        current_correct = (targets == predictions).sum()
        current_result = careful_divide(current_correct, current_total)

        self.correct += current_correct
        self.total += current_total

        return current_result

    def summary(self) -> float:
        return careful_divide(self.correct, self.total)

    def name(self) -> str:
        if self.ignore_ids:
            return "MoF-BG"
        else:
            return "MoF"


class F1Score(object):
    def __init__(
        self,
        overlaps: Iterable[float] = (0.1, 0.25, 0.5),
        ignore_ids: List[int] = [],
        num_classes: int = None,
    ):
        self.overlaps = overlaps
        self.ignore_ids = ignore_ids
        self.num_classes = num_classes
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        if self.num_classes is None:
            shape = (len(self.overlaps), 1)
        else:
            shape = (len(self.overlaps), self.num_classes)
        self.tp = np.zeros(shape)
        self.fp = np.zeros(shape)
        self.fn = np.zeros(shape)

    def f_score(self, recognized: List[int], ground_truth: List[int], overlap: float):
        p_label, p_start, p_end = get_labels_start_end_time(recognized, self.ignore_ids)
        y_label, y_start, y_end = get_labels_start_end_time(ground_truth, self.ignore_ids)
        tp = 0
        fp = 0
        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)

    def add(
            self,
            targets: Union[List[int], List[List[int]]],
            predictions: Union[List[int], List[List[int]]]
    ) -> dict:
        """

        Args:
            targets: array of size [T, C]  or [T]
            predictions: array of size [T, C]  or [T]

        Returns:

        """
        current_result = {}
        targets = np.array(targets)
        predictions = np.array(predictions)
        if targets.ndim == 1:
            targets = targets.reshape((-1, 1))
        if predictions.ndim == 1:
            predictions = predictions.reshape((-1, 1))
        f1_per_class = []

        num_classes = self.num_classes if self.num_classes else 1
        for s in range(len(self.overlaps)):
            for c_idx in range(num_classes):
                tp1, fp1, fn1 = self.f_score(
                    predictions[:, c_idx],
                    targets[:, c_idx],
                    self.overlaps[s],
                )
                self.tp[s, :] += tp1  # [num_overlaps, num_classes]
                self.fp[s, :] += fp1
                self.fn[s, :] += fn1

                current_f1 = self.get_f1_score(tp1, fp1, fn1)   # [C]
                f1_per_class.append(current_f1)
            current_result[f"F1@{int(self.overlaps[s]*100)}"] = np.mean(f1_per_class)
        return current_result

    def summary(self) -> dict:
        result = {}
        for s in range(len(self.overlaps)):
            f1_per_class = self.get_vectorized_f1(tp=self.tp[s], fp=self.fp[s], fn=self.fn[s])
            result[f"F1@{int(self.overlaps[s]*100)}"] = np.mean(f1_per_class)

        return result

    @staticmethod
    def get_vectorized_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        """
        Args:
            tp: [num_classes]
            fp: [num_classes]
            fn: [num_classes]
        Returns:
            [num_classes]
        """
        return 2 * tp / (2 * tp + fp + fn + 0.00001)

    @staticmethod
    def get_f1_score(tp: float, fp: float, fn: float) -> float:
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0

        if precision + recall != 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = f1 * 100
        else:
            f1 = 0.0

        return f1


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self):
        self.metric_dict = {
            "MoF": MoFAccuracyMetric(),
            "Edit": Edit(),
            "F1": F1Score(),
        }
        self.num_samples = 0
        self.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        for metric in self.metric_dict.values():
            metric.reset()
        self.num_samples = 0

    def update_stats(self, target, prediction, num_videos=1):
        for name, metric in self.metric_dict.items():
            metric.add(list(target), list(prediction))

        self.num_samples += num_videos

    def log_stats(self):
        """
        Log the calculated metrics.

        Returns:
            dict: dictionary which contains the metrics (MoF, Edit, F1@10, F1@25, F1@50)
        """
        stats = {}
        for name, metric in self.metric_dict.items():
            result = metric.summary()
            if isinstance(result, dict):
                for n, v in result.items():
                    stats[n] = v
            else:
                stats[name] = result
        return stats
