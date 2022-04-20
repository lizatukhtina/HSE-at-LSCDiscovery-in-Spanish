import numpy as np
import os
import pickle
from abc import abstractmethod, ABC
from math import inf
from scipy import stats
from torch.nn import CosineSimilarity
from scipy.spatial.distance import cdist, cosine
from sklearn.cluster import KMeans, AffinityPropagation
from typing import Dict, List

from argument_parser import Argument, CustomArgumentParser, EnumArgument, OptionalArgument


class Score:
    def __init__(self, graded: float = 0, binary: int = 0, gain: int = 0, loss: int = 0):
        self.graded = graded
        self.binary = binary
        self.gain = gain
        self.loss = loss


class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_metric(self, emb1, emb2):
        pass


class MetricAverage(Metric):
    def __init__(self):
        super().__init__()

    def get_metric(self, emb1, emb2):
        try:
            average_first = emb1.mean(axis=0)
            average_second = emb2.mean(axis=0)
            try:
                cosine_change = float(1 - CosineSimilarity(dim=0)(average_first, average_second))
            except TypeError:
                cosine_change = cosine(average_first, average_second)
            return Score(graded=cosine_change)
        except ValueError:
            return None


class MetricKMeans(Metric):
    def __init__(self, n_clusters: int):
        super().__init__()
        self.n_clusters = n_clusters

    def get_cluster_centroids(self, word_embeddings):
        if len(word_embeddings) > self.n_clusters:
            clustering = KMeans(random_state=0, n_clusters=self.n_clusters).fit(word_embeddings)
            centroids = clustering.cluster_centers_
        else:
            centroids = word_embeddings
        return centroids

    def get_metric(self, emb1, emb2):
        try:
            centroids_first = self.get_cluster_centroids(emb1)
            centroids_second = self.get_cluster_centroids(emb2)
            cosine_change = cdist(centroids_first, centroids_second)
            mean = np.mean(cosine_change)
            n_clusters1 = len(MetricAffProp().get_cluster_centers(centroids_first))
            n_clusters2 = len(MetricAffProp().get_cluster_centers(centroids_second))
            gain = 1 if n_clusters2 > n_clusters1 else 0
            loss = 1 if n_clusters1 > n_clusters2 else 0
            return Score(graded=float(mean), gain=gain, loss=loss)
        except ValueError:
            return None

    def get_cluster_labels(self, scores):
        clustering = KMeans(random_state=0, n_clusters=self.n_clusters).fit(scores)
        if float(clustering.cluster_centers_[0]) > float(clustering.cluster_centers_[1]):
            return [1 - i for i in clustering.labels_]
        else:
            return [i for i in clustering.labels_]


class MetricAffProp(Metric):
    def __init__(self):
        super().__init__()

    def get_metric(self, emb1, emb2):
        pass

    @staticmethod
    def get_cluster_centers(scores):
        clustering = AffinityPropagation(random_state=0, max_iter=4000).fit(scores)
        return clustering.cluster_centers_

    @staticmethod
    def get_cluster_labels(scores):
        clustering = AffinityPropagation(random_state=0, max_iter=4000).fit(scores)
        average = clustering.cluster_centers_.mean()
        result = []
        for label in clustering.labels_:
            if clustering.cluster_centers_[label] > average:
                result.append(1)
            else:
                result.append(0)
        return result


class Dispatcher:
    def __init__(self, embeddings_path: str, gold_standard_path: str, eval_metric: Metric, report_path: str):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        with open(gold_standard_path, encoding='utf-8-sig') as f:
            gold_standard = [i.split() for i in f.readlines()]
        try:
            scores = {i[0]: i[1] for i in gold_standard}
            self.sub_worker = Comparator(embeddings=embeddings, eval_metric=eval_metric, gold_standard=scores,
                                         report_path=report_path)
        except IndexError:
            self.sub_worker = Grader(embeddings=embeddings, eval_metric=metric,
                                     gold_standard=[i[0] for i in gold_standard], report_path=report_path)

    def evaluate(self):
        self.sub_worker.evaluate()


class Worker(ABC):
    def __init__(self, embeddings, gold_standard, eval_metric: Metric, report_path):
        self.embeddings = embeddings
        self.metric = eval_metric
        self.gold_standard = gold_standard
        self.report_path = report_path

    def evaluate(self):
        all_metrics = []
        words = []
        time_epochs = list(self.embeddings.keys())
        for i, word in enumerate(self.gold_standard):
            print(f'Processing word {word} ({i + 1} out of {len(self.gold_standard)})')
            if word not in self.embeddings[time_epochs[0]] and word not in self.embeddings[time_epochs[1]]:
                print('ERROR: word is not present in any of the slices')
                continue
            if word not in self.embeddings[time_epochs[0]]:
                print('WARNING: Word not present in the first slice')
                all_metrics.append(Score(graded=inf, binary=1, gain=1, loss=0))
                words.append(word)
            elif word not in self.embeddings[time_epochs[1]]:
                print('WARNING: Word not present in the seconds slice')
                all_metrics.append(Score(graded=inf, binary=1, gain=0, loss=1))
                words.append(word)
            else:
                word_metrics = self.metric.get_metric(self.embeddings[time_epochs[0]][word],
                                                      self.embeddings[time_epochs[1]][word])
                if not word_metrics:
                    print('ERROR: Failed to evaluate metric')
                    continue
                all_metrics.append(word_metrics)
                words.append(word)
                print(f'{word} - {word_metrics.graded}')
        max_graded = max(i.graded for i in all_metrics if i.graded != inf)
        all_metrics = [Score(graded=max_graded, binary=i.binary, gain=i.gain, loss=i.loss)
                       if i.graded == inf else i for i in all_metrics]
        binary_grader = MetricKMeans(n_clusters=2)
        binary_scores = binary_grader.get_cluster_labels([[i.graded] for i in all_metrics])
        normalized_means = Worker.normalize([i.graded for i in all_metrics])
        final_scores = {word: Score(graded=graded, binary=binary, gain=init_metric.gain, loss=init_metric.loss) for
                        word, graded, binary, init_metric in zip(words, normalized_means, binary_scores, all_metrics)}
        self.print_report(final_scores)

    @staticmethod
    def normalize(data: list) -> list:
        normalized = [(x - min(data)) / (max(data) - min(data)) for x in data]
        return normalized

    @abstractmethod
    def print_report(self, scores):
        pass


class Grader(Worker):
    def __init__(self, embeddings, gold_standard, eval_metric: Metric, report_path):
        super().__init__(embeddings, gold_standard, eval_metric, report_path)

    def print_report(self, scores: Dict[List, Score]):
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write('word\tchange_binary\tchange_binary_gain\tchange_binary_loss\tchange_graded\tCOMPARE\n')
            for word in self.gold_standard:
                if word not in scores:
                    scores[word] = Score(graded=0.5, binary=0)
                print(f'{word}: graded = {scores[word].graded}; binary = {scores[word].binary}; '
                      f'gain = {scores[word].gain}; loss = {scores[word].loss}')
                f.write(f'{word}\t{scores[word].binary}\t{scores[word].gain}\t{scores[word].loss}\t'
                        f'{scores[word].graded}\t{scores[word].graded}\n')


class Comparator(Worker):
    def __init__(self, embeddings, gold_standard, eval_metric: Metric, report_path: str):
        super().__init__(embeddings, gold_standard, eval_metric, report_path)

    def print_report(self, scores: Dict[List, Score]):
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            x = []
            y = []
            for word in self.gold_standard:
                if word not in scores:
                    scores[word].graded = 0.5
                    x.append(0.5)
                else:
                    x.append(scores[word].graded)
                y.append(self.gold_standard[word])
                print(f'{word} - {scores[word].graded} - {self.gold_standard[word]}')
                f.write(f'{word} - {scores[word].graded} - {self.gold_standard[word]}\n')
            print(str(stats.spearmanr(a=x, b=y)))
            f.write(str(stats.spearmanr(a=x, b=y)))


arg_list = [Argument(name='embeddings_path', type_=str, help_message='Path to a file with embeddings',
                     absent_message='Please specify a path to the file with embeddings'),
            Argument(name='targets', type_=str, help_message='Path to a file with target words',
                     absent_message='Please specify a path to target words'),
            EnumArgument(name='metric', type_=str, help_message='Metric to evaluate embeddings',
                         absent_message='Please specify metric for evaluating embeddings',
                         choices=['average', 'k_means']),
            Argument(name='report_path', type_=str, help_message='Path for writing the report file',
                     absent_message='Please specify a path for writing the report file'),
            OptionalArgument(name='n_clusters', type_=int, help_message='Number of clusters (for k_means)')]
arg_parser = CustomArgumentParser()
args = arg_parser.parse(arg_list)
metric = None
if args.metric == 'average':
    metric = MetricAverage()
elif 'k_means' in args.metric:
    clusters = 8
    try:
        clusters = args.n_clusters
    except AttributeError:
        pass
    metric = MetricKMeans(n_clusters=clusters)
dispatcher = Dispatcher(embeddings_path=args.embeddings_path, gold_standard_path=args.targets,
                        eval_metric=metric, report_path=args.report_path)
dispatcher.evaluate()
