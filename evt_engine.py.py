import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans

class EVT_Engine_V2:
    def __init__(self, tail_ratio=0.10, min_tail=5, max_tail=25, distance_metric='cosine',
                 temperature=1.5, max_prototypes=3, min_samples_per_proto=15,
                 threshold_quantile=0.05, margin_quantile=0.05, fallback_min_correct=6):
        self.tail_ratio = tail_ratio
        self.min_tail = min_tail
        self.max_tail = max_tail
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.max_prototypes = max_prototypes
        self.min_samples_per_proto = min_samples_per_proto
        self.threshold_quantile = threshold_quantile
        self.margin_quantile = margin_quantile
        self.fallback_min_correct = fallback_min_correct

        self.class_models = {}
        self.class_thresholds = {}
        self.global_threshold = 0.0
        self.margin_threshold = 0.0
        self.known_classes = []

    def _l2_normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            norm = np.linalg.norm(x) + 1e-12
            return x / norm
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norm

    def _softmax(self, logits):
        logits = np.asarray(logits, dtype=np.float64) / max(self.temperature, 1e-8)
        logits = logits - np.max(logits)
        expv = np.exp(logits)
        return expv / (np.sum(expv) + 1e-12)

    def calc_distance(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if self.distance_metric == 'euclidean':
            return float(np.linalg.norm(x - y))
        elif self.distance_metric == 'cosine':
            sim = np.clip(np.dot(x, y), -1.0, 1.0)
            return float(1.0 - sim)
        else:
            raise ValueError(f"Unsupported metric: {self.distance_metric}")

    def _decide_num_prototypes(self, n):
        if n < self.min_samples_per_proto * 2:
            return 1
        k = n // self.min_samples_per_proto
        k = max(1, min(k, self.max_prototypes))
        return k

    def _fit_tail_model(self, distances):
        distances = np.sort(np.asarray(distances, dtype=np.float64))
        n = len(distances)
        if n == 0:
            return None

        tail_count = int(round(n * self.tail_ratio))
        tail_count = min(n, max(self.min_tail, min(self.max_tail, tail_count)))
        tail_raw = distances[-tail_count:]
        u = float(tail_raw[0])
        excess = tail_raw - u
        pos_excess = excess[excess > 1e-12]

        if len(pos_excess) >= 2:
            try:
                shape, _, scale = stats.weibull_min.fit(pos_excess, floc=0)
                scale, shape = max(float(scale), 1e-6), float(shape)
            except:
                shape, scale = 1.0, max(float(np.std(tail_raw)), 1e-3)
        else:
            shape, scale = 1.0, max(float(np.std(tail_raw)), 1e-3)

        return {'u': u, 'shape': shape, 'scale': scale, 'tail_count': tail_count,
                'max_dist': float(distances[-1]), 'mean_dist': float(np.mean(distances))}

    def _prototype_inlier_score(self, feature, proto_model):
        d = self.calc_distance(feature, proto_model['center'])
        u = max(proto_model['u'], 1e-8)
        if d <= u:
            score = np.exp(-d / u)
        else:
            prob_outlier = stats.weibull_min.cdf(d - u, proto_model['shape'], loc=0, scale=max(proto_model['scale'], 1e-8))
            score = np.exp(-1.0) * (1.0 - prob_outlier)
        return float(np.clip(score, 0.0, 1.0))

    def _class_evt_score(self, feature, class_id):
        proto_list = self.class_models.get(class_id, [])
        if not proto_list: return 0.0
        return float(np.max([self._prototype_inlier_score(feature, proto) for proto in proto_list]))

    def _joint_scores_from_normalized_feature(self, feature_norm, logits):
        probs = self._softmax(logits)
        evt_scores = np.zeros(len(probs), dtype=np.float32)
        joint_scores = np.zeros(len(probs), dtype=np.float32)

        for c in self.known_classes:
            evt_c = self._class_evt_score(feature_norm, c)
            evt_scores[c] = evt_c
            joint_scores[c] = float(probs[c] * evt_c)

        pred = int(np.argmax(joint_scores))
        sorted_idx = np.argsort(joint_scores)[::-1]
        top1 = float(joint_scores[sorted_idx[0]])
        margin = top1 - float(joint_scores[sorted_idx[1]]) if len(sorted_idx) > 1 else top1

        return pred, top1, margin, probs, evt_scores, joint_scores

    def fit(self, features, logits, labels, known_classes, logger):
        self.known_classes = sorted(list(known_classes))
        features_norm = self._l2_normalize(features)
        preds_cls = np.argmax(logits, axis=1)
        correct_mask = preds_cls == labels

        for c in self.known_classes:
            class_correct = features_norm[(labels == c) & correct_mask]
            class_feats = class_correct if len(class_correct) >= self.fallback_min_correct else features_norm[(labels == c)]
            if len(class_feats) == 0: continue

            k = max(1, min(self._decide_num_prototypes(len(class_feats)), len(class_feats)))
            if k == 1:
                assignments, centers = np.zeros(len(class_feats), dtype=np.int64), np.mean(class_feats, axis=0, keepdims=True)
            else:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    assignments, centers = kmeans.fit_predict(class_feats), kmeans.cluster_centers_
                except:
                    assignments, centers, k = np.zeros(len(class_feats), dtype=np.int64), np.mean(class_feats, axis=0, keepdims=True), 1

            proto_models = []
            for pid in range(k):
                proto_feats = class_feats[assignments == pid]
                if len(proto_feats) == 0: continue
                center = self._l2_normalize(np.mean(proto_feats, axis=0))
                dists = [self.calc_distance(f, center) for f in proto_feats]
                tail_model = self._fit_tail_model(dists)
                if tail_model:
                    proto_models.append({
                        'center': center.astype(np.float32), 'u': tail_model['u'],
                        'shape': tail_model['shape'], 'scale': tail_model['scale']
                    })
            if proto_models:
                self.class_models[c] = proto_models

        class_score_buckets = {c: [] for c in self.known_classes}
        margin_list = []
        for feat_norm, logit, label in zip(features_norm, logits, labels):
            pred, conf, margin, _, _, _ = self._joint_scores_from_normalized_feature(feat_norm, logit)
            if pred == label:
                class_score_buckets[label].append(conf)
                margin_list.append(margin)

        all_valid_scores = []
        for c in self.known_classes:
            scores_c = class_score_buckets[c]
            if scores_c:
                self.class_thresholds[c] = max(float(np.quantile(scores_c, self.threshold_quantile)), 1e-8)
                all_valid_scores.extend(scores_c)
            else:
                self.class_thresholds[c] = None

        self.global_threshold = float(np.quantile(all_valid_scores, self.threshold_quantile)) if all_valid_scores else 0.05
        for c in self.known_classes:
            if self.class_thresholds[c] is None:
                self.class_thresholds[c] = self.global_threshold

        self.margin_threshold = float(np.quantile(margin_list, self.margin_quantile)) if margin_list else 0.0
        logger.log("[EVT-V2] Calibration finished successfully.")

    def predict_open_set_score(self, feature, logits):
        if not self.class_models:
            return int(np.argmax(logits)), 0.0, False, {'margin': 0.0, 'class_threshold': self.global_threshold}

        feat_norm = self._l2_normalize(feature)
        pred_cls, conf, margin, probs, evt_scores, joint_scores = self._joint_scores_from_normalized_feature(feat_norm, logits)

        class_tau = self.class_thresholds.get(pred_cls, self.global_threshold)
        accepted = (conf >= class_tau) and (margin >= self.margin_threshold)

        return pred_cls, float(conf), bool(accepted), {
            'margin': float(margin), 'class_threshold': float(class_tau),
            'prob_pred': float(probs[pred_cls]) if pred_cls < len(probs) else 0.0,
            'evt_pred': float(evt_scores[pred_cls]) if pred_cls < len(evt_scores) else 0.0,
            'joint_pred': float(joint_scores[pred_cls]) if pred_cls < len(joint_scores) else 0.0
        }