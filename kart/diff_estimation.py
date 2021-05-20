import itertools
import logging
import statistics
import subprocess
import time
from threading import Thread
from queue import Queue, Empty

FEATURE_SUBTREES_PER_TREE = 256
FEATURE_TREE_NESTING = 2
MAX_TREES = FEATURE_SUBTREES_PER_TREE ** FEATURE_TREE_NESTING

EMPTY_SHA = "0" * 40
Z_SCORES = {
    0.50: 0.0,
    0.60: 0.26,
    0.70: 0.53,
    0.75: 0.68,
    0.80: 0.85,
    0.85: 1.04,
    0.90: 1.29,
    0.95: 1.65,
    0.99: 2.33,
}

L = logging.getLogger("kart.diff_estimation")


class BaseEstimator:
    def __init__(self, repo):
        self.repo = repo

    def _get_feature_sample_paths(self, feature_path, num_trees):
        """
        Returns a list of tree paths which cover the given number of feature trees.
        """
        num_full_subtrees = num_trees // 256
        paths = [f"{feature_path}{n:02x}" for n in range(num_full_subtrees)]
        paths.extend(
            [
                f"{feature_path}{num_full_subtrees:02x}/{n:02x}"
                for n in range(num_trees % 256)
            ]
        )
        return paths

    def _git_diff_tree(self, tree_id1, tree_id2, feature_path, num_trees, *extra_args):
        paths = self._get_feature_sample_paths(feature_path, num_trees)
        p = subprocess.Popen(
            [
                "git",
                "-C",
                str(self.repo.path),
                "diff-tree",
                "-r",
                tree_id1.hex,
                tree_id2.hex,
                *extra_args,
                "--",
                *paths,
            ],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        return p

    def get_annotation_type(self, accuracy):
        raise NotImplementedError

    def _get_dataset_estimate(
        self,
        base_rs,
        target_rs,
        dataset_path,
        *,
        working_copy,
        accuracy,
    ):
        raise NotImplementedError

    def get_estimate(
        self,
        base_rs,
        target_rs,
        *,
        working_copy=None,
        accuracy,
    ):
        """
        Estimates feature counts for each dataset in the given diff.
        Returns a dict (keys are dataset paths; values are feature counts)
        Datasets with (probably) no features changed are not present in the dict.
        `accuracy` should be one of self.ACCURACY_CHOICES
        """
        if base_rs == target_rs and not working_copy:
            return {}

        assert accuracy in self.ACCURACY_CHOICES
        assert base_rs.repo == target_rs.repo == self.repo
        base_ds_paths = {ds.path for ds in base_rs.datasets}
        target_ds_paths = {ds.path for ds in target_rs.datasets}
        all_ds_paths = base_ds_paths | target_ds_paths

        annotation_type = self.get_annotation_type(accuracy)
        annotation = self.repo.diff_annotations.get(
            base_rs=base_rs,
            target_rs=target_rs,
            annotation_type=annotation_type,
        )
        if annotation is not None:
            return annotation

        result = {}
        for dataset_path in all_ds_paths:
            ds_result = self._get_dataset_estimate(
                base_rs,
                target_rs,
                dataset_path,
                working_copy=working_copy,
                accuracy=accuracy,
            )

            if ds_result:
                result[dataset_path] = ds_result

        if not working_copy:
            self.repo.diff_annotations.store(
                base_rs=base_rs,
                target_rs=target_rs,
                annotation_type=annotation_type,
                data=result,
            )

        return result

    def _estimate(self):
        raise NotImplementedError


class FeatureCountEstimator(BaseEstimator):
    ACCURACY_CHOICES = ("veryfast", "fast", "medium", "good", "exact")

    def get_annotation_type(self, accuracy):
        return f"feature-change-counts-{accuracy}"

    def _feature_count_sample_trees(self, tree_id1, tree_id2, feature_path, num_trees):
        p = self._git_diff_tree(tree_id1, tree_id2, feature_path, num_trees)
        tree_samples = {}
        for line in p.stdout:
            # path/to/dataset/.sno-dataset/feature/ab/cd/abcdef123
            # --> ab/cd
            root, tree, subtree, basename = line.rsplit("/", 3)
            k = f"{tree}/{subtree}"
            tree_samples.setdefault(k, 0)
            tree_samples[k] += 1
        p.wait()
        r = list(tree_samples.values())
        r.extend([0] * (num_trees - len(r)))
        return r

    def _get_dataset_estimate(
        self,
        base_rs,
        target_rs,
        dataset_path,
        *,
        working_copy,
        accuracy,
    ):
        """
        Estimates feature counts for each dataset in the given diff.
        Returns a dict (keys are dataset paths; values are feature counts)
        Datasets with (probably) no features changed are not present in the dict.
        `accuracy` should be one of ACCURACY_CHOICES
        """
        if accuracy == "exact" and working_copy:
            # can't really avoid this - to generate an exact count for this diff we have to generate the diff
            from kart.diff import get_dataset_diff

            ds_diff = get_dataset_diff(
                base_rs,
                target_rs,
                working_copy,
                dataset_path,
            )
            if "feature" not in ds_diff:
                ds_total = 0
            else:
                ds_total = len(ds_diff["feature"])
            return ds_total

        base_ds = base_rs.datasets.get(dataset_path)
        target_ds = target_rs.datasets.get(dataset_path)

        if not base_ds:
            base_ds, target_ds = target_ds, base_ds

        # Come up with a list of trees to diff.
        feature_path = (
            f"{base_ds.path}/{base_ds.DATASET_DIRNAME}/{base_ds.FEATURE_PATH}"
        )
        ds_total = 0
        if (not target_ds) or base_ds.feature_tree != target_ds.feature_tree:
            if accuracy == "exact":
                ds_total += sum(
                    self._feature_count_sample_trees(
                        base_rs.tree.id,
                        target_rs.tree.id,
                        feature_path,
                        MAX_TREES,
                    )
                )
            else:
                if accuracy == "veryfast":
                    # only ever sample two trees
                    sample_size = 2
                    required_confidence = 0.00001
                    z_score = 0.0
                else:
                    if accuracy == "fast":
                        sample_size = 2
                        required_confidence = 0.60
                    elif accuracy == "medium":
                        sample_size = 8
                        required_confidence = 0.80
                    elif accuracy == "good":
                        sample_size = 16
                        required_confidence = 0.95
                    z_score = Z_SCORES[required_confidence]

                sample_mean = 0
                while sample_size <= MAX_TREES:
                    L.debug(
                        "sampling %d trees for dataset %s",
                        sample_size,
                        dataset_path,
                    )
                    t1 = time.monotonic()
                    samples = self._feature_count_sample_trees(
                        base_rs.tree.id,
                        target_rs.tree.id,
                        feature_path,
                        sample_size,
                    )
                    sample_mean = statistics.mean(samples)
                    sample_stdev = statistics.stdev(samples)

                    t2 = time.monotonic()
                    if accuracy == "veryfast":
                        # even if no features were found in the two trees, call it done.
                        # this will be Good Enough if all you need to know is something like
                        # "is the diff size probably less than 100K features?"
                        break
                    if sample_mean == 0:
                        # no features were encountered in the sample.
                        # this is likely quite a small diff.
                        # let's just sample a lot more trees.
                        new_sample_size = sample_size * 1024
                        if new_sample_size > MAX_TREES:
                            L.debug(
                                "sampled %s trees in %.3fs, found 0 features; stopping",
                                sample_size,
                                t2 - t1,
                            )
                        else:
                            L.debug(
                                "sampled %s trees in %.3fs, found 0 features; increased sample size to %d",
                                sample_size,
                                t2 - t1,
                                new_sample_size,
                            )
                        sample_size = new_sample_size
                        continue

                    # try and get within 10% of the real mean.
                    margin_of_error = 0.10 * sample_mean
                    required_sample_size = min(
                        MAX_TREES,
                        (z_score * sample_stdev / margin_of_error) ** 2,
                    )
                    L.debug(
                        "sampled %s trees in %.3fs (ƛ=%.3f, s=%.3f). required: %.1f (margin: %.1f; confidence: %d%%)",
                        sample_size,
                        t2 - t1,
                        sample_mean,
                        sample_stdev,
                        required_sample_size,
                        margin_of_error * MAX_TREES,
                        required_confidence * 100,
                    )
                    if sample_size >= required_sample_size:
                        break

                    if sample_size == MAX_TREES:
                        break
                    while sample_size < required_sample_size:
                        sample_size *= 2
                    sample_size = min(MAX_TREES, sample_size)
                ds_total += int(round(sample_mean * MAX_TREES))

        if working_copy:
            ds_total += working_copy.tracking_changes_count(base_ds)
        return ds_total


class NonBlockingStreamReader:
    """
    Starts a thread which reads from a stream and puts each line into a queue.
    """

    def __init__(self, stream):
        self._s = stream
        self._q = Queue()
        self._finished = False

        def populate_queue(stream, queue):
            for line in stream:
                self._q.put(line)
            self._finished = True

        self._t = Thread(target=populate_queue, args=(self._s, self._q))
        self._t.daemon = True
        self._t.start()  # start collecting lines from the stream

    def readline(self):
        while True:
            try:
                return self._q.get(block=True, timeout=0.05)
            except Empty:
                if self._finished:
                    return None

    def __next__(self):
        x = self.readline()
        if x is None:
            raise StopIteration
        else:
            return x

    def __iter__(self):
        return self


class BlobSizeSummer:
    """
    Given a bunch of blob IDs, sums blob size asynchronously,
    by piping their IDs via git cat-file
    """

    def __init__(self):
        self._popen = subprocess.Popen(
            ["git", "cat-file", "--batch-check=%(objectsize)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            encoding="ascii",
        )
        self._nbsr = NonBlockingStreamReader(self._popen.stdout)

    def add_blob(self, blob_id):
        self._popen.stdin.write(f"{blob_id}\n")

    def total(self):
        self._popen.stdin.close()
        self._popen.wait()
        t = 0
        for line in self._nbsr:
            t += int(line[:-1])
        return t


class TotalFeatureSizeEstimator(BaseEstimator):
    ACCURACY_CHOICES = ("veryfast", "fast", "medium", "good")

    def get_annotation_type(self, accuracy):
        return f"total-feature-size-{accuracy}"

    def _get_diff_blob_ids(self, diff_tree_output):
        """
        Given the stream of output from git-diff-tree,
        yields 2-tuples of all blob IDs that have changed along with their subtree.
        e.g.
            ("00/00", "abcdef1234557890abcdef1234557890abcdef12")
        """
        for line in diff_tree_output:
            sha1 = line[15:55]
            sha2 = line[56:96]
            path = line[100:]
            root, tree, subtree, basename = path.rsplit("/", 3)
            k = f"{tree}/{subtree}"
            if sha1 != EMPTY_SHA:
                yield k, sha1
            if sha2 != EMPTY_SHA:
                yield k, sha2

    def _blob_size_sample_trees(self, tree_id1, tree_id2, feature_path, num_trees):
        # call git-diff-tree to generate blob IDs
        p = self._git_diff_tree(tree_id1, tree_id2, feature_path, num_trees)
        blob_ids_iter = self._get_diff_blob_ids(p.stdout)

        blob_size_samples = []
        num_blobs_samples = []
        # segment the blob IDs into samples based on subtree
        for subtree, blob_ids in itertools.groupby(blob_ids_iter, key=lambda t: t[0]):
            # for each sample, sum all the blob sizes
            summer = BlobSizeSummer()
            num_blobs = 0
            for _, blob_id in blob_ids:
                summer.add_blob(blob_id)
                num_blobs += 1
            blob_size_samples.append(summer.total())
            num_blobs_samples.append(num_blobs)
        # if any samples had no blobs, add in the 0 samples
        blob_size_samples.extend([0] * (num_trees - len(blob_size_samples)))
        num_blobs_samples.extend([0] * (num_trees - len(num_blobs_samples)))
        return blob_size_samples, num_blobs_samples

    def _get_dataset_estimate(
        self,
        base_rs,
        target_rs,
        dataset_path,
        *,
        working_copy,
        accuracy,
    ):
        base_ds = base_rs.datasets.get(dataset_path)
        target_ds = target_rs.datasets.get(dataset_path)

        if not base_ds:
            base_ds, target_ds = target_ds, base_ds

        # Come up with a list of trees to diff.
        feature_path = (
            f"{base_ds.path}/{base_ds.DATASET_DIRNAME}/{base_ds.FEATURE_PATH}"
        )
        ds_total = 0
        mean_blob_size = 0
        if (not target_ds) or base_ds.feature_tree != target_ds.feature_tree:
            if accuracy == "veryfast":
                # only ever sample two trees
                sample_size = 2
                required_confidence = 0.00001
                z_score = 0.0
            else:
                if accuracy == "fast":
                    sample_size = 8
                    required_confidence = 0.60
                elif accuracy == "medium":
                    sample_size = 16
                    required_confidence = 0.70
                elif accuracy == "good":
                    sample_size = 64
                    required_confidence = 0.85
                z_score = Z_SCORES[required_confidence]

            sample_mean = 0
            while sample_size <= MAX_TREES:
                L.debug(
                    "sampling %d trees for dataset %s",
                    sample_size,
                    dataset_path,
                )
                t1 = time.monotonic()
                blob_size_samples, num_blobs_samples = self._blob_size_sample_trees(
                    base_rs.tree.id,
                    target_rs.tree.id,
                    feature_path,
                    sample_size,
                )
                sample_mean = statistics.mean(blob_size_samples)
                sample_stdev = statistics.stdev(blob_size_samples)
                if any(num_blobs_samples):
                    mean_blob_size = statistics.mean(
                        [
                            (size / count)
                            for (size, count) in zip(
                                blob_size_samples, num_blobs_samples
                            )
                            if count
                        ]
                    )

                t2 = time.monotonic()
                if accuracy == "veryfast":
                    # even if no features were found in the two trees, call it done.
                    # this will be Good Enough if all you need to know is something like
                    # "is the diff size probably less than 100 MB?"
                    break
                if sample_mean == 0:
                    # no features were encountered in the sample.
                    # this is likely quite a small diff.
                    # let's just sample a lot more trees.
                    new_sample_size = sample_size * 64
                    if new_sample_size > MAX_TREES:
                        L.debug(
                            "sampled %s trees in %.3fs, found 0 features; stopping",
                            sample_size,
                            t2 - t1,
                        )
                    else:
                        L.debug(
                            "sampled %s trees in %.3fs, found 0 features; increased sample size to %d",
                            sample_size,
                            t2 - t1,
                            new_sample_size,
                        )
                    sample_size = new_sample_size
                    continue

                # try and get within 50% of the real mean.
                margin_of_error = 0.50 * sample_mean
                required_sample_size = min(
                    MAX_TREES,
                    (z_score * sample_stdev / margin_of_error) ** 2,
                )
                L.debug(
                    "sampled %s trees in %.3fs (ƛ=%.3f, s=%.3f). required: %.1f (margin: %.1f; confidence: %d%%)",
                    sample_size,
                    t2 - t1,
                    sample_mean,
                    sample_stdev,
                    required_sample_size,
                    margin_of_error * MAX_TREES,
                    required_confidence * 100,
                )
                if sample_size >= required_sample_size:
                    break

                if sample_size == MAX_TREES:
                    break
                while sample_size < required_sample_size:
                    sample_size *= 2
                sample_size = min(MAX_TREES, sample_size)
            ds_total += int(round(sample_mean * MAX_TREES))

        if working_copy:
            ds_total += mean_blob_size * working_copy.tracking_changes_count(base_ds)
        return ds_total
