from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelmax, argrelmin
from sklearn.preprocessing import minmax_scale

from binning import RDdata


def init_segment(data: np.ndarray):
    """Segmentation of data

    Parameters
    ----------
    data: np.ndarray
    One-dimensional input data

    Returns
    -------
    C: calculated value representing whether point is a breakpoint
    start_idx: shape(N), the start index of each segment
    """
    data = minmax_scale(data.reshape(-1, 1)).ravel()
    limit, thres, box = 60, 0.5, 3.5
    tmp_std = data.std() ** 2
    if tmp_std > 0.0005:
        limit = int(6000 * tmp_std + 30)
        thres = round(1 - 50 * tmp_std, 1)
        box = round(3.8 - 100 * tmp_std, 1)
        limit = min(80, limit)
        thres = max(0.3, min(0.7, thres))
        box = max(3, box)

    # calculate C
    w = np.apply_along_axis(
        lambda x: np.exp(-x * x), 0, np.diff(data, prepend=data[-1])
    )
    C = np.zeros(len(data))
    for i in range(limit, len(data) - limit):
        left_w = right_w = w[i]
        left_wx = left_w * data[i - 1]
        right_wx = right_w * data[i]

        tmp_w = left_w
        for k in range(i - 2, i - limit - 1, -1):
            tmp_w *= w[k + 1]
            if i - k >= 8 and tmp_w < thres:
                break
            left_w += tmp_w
            left_wx += tmp_w * data[k]
        tmp_w = right_w
        for k in range(i + 1 - len(data), i + limit - len(data)):
            tmp_w *= w[k]
            if k + len(data) - i >= 8 and tmp_w < thres:
                break
            right_w += tmp_w
            right_wx += tmp_w * data[k]

        C[i] = right_wx / right_w - left_wx / left_w

    # select change point by threshold => [start,end)
    Q1 = np.quantile(C, 0.25)
    Q3 = np.quantile(C, 0.75)
    IQR = Q3 - Q1
    lower = Q1 - box * IQR
    upper = Q3 + box * IQR
    idx = np.select([C < lower, C > upper], [-1, 1], 0)
    idx[0] = idx[-1] = 0
    change_point = [0]  # store start point (contain) idx of each segment
    start = 0
    for i, ty in enumerate(idx):
        if ty == idx[start]:
            continue
        if start == 0 or idx[start] == 0:
            start = i
            continue
        num = (i - start) // 10
        num = num if num > 1 else 1
        if idx[start] == 1:
            p = argrelmax(C[start - 1 : i + 1])[0] - 1 + start
            p = p[np.argsort(C[p])[-num:]]
        else:
            p = argrelmin(C[start - 1 : i + 1])[0] - 1 + start
            p = p[np.argsort(C[p])[:num]]
        start = i
        change_point.extend(np.sort(p))
    # delete segment that only have one point
    for i in range(len(change_point) - 1, 0, -1):
        cur = change_point[i]
        if cur - 1 <= change_point[i - 1]:
            if (
                cur >= len(data) - 1
                or data[cur] - data[cur - 1] <= data[cur] - data[cur + 1]
            ):
                change_point[i - 1] = cur
            del change_point[i]
    start_idx = np.asarray(change_point)

    return C, start_idx


def rearrange_segment_value(data: np.ndarray, start_idx: np.ndarray):
    """Segmentation and value adjustment of data

    Parameters
    ----------
    data: one-dimensional input data
    start_idx: shape(N), the start index of each segment

    Returns
    -------
    segment: array, shape(N, 2), like [[start, end),...]
    u: array, representative value of each segment
    """
    # init mean, var, length
    sub_seg = np.split(data, start_idx[1:])
    u = np.fromiter((np.mean(a) for a in sub_seg), dtype=np.float64)
    s = np.fromiter((np.var(a, ddof=1) for a in sub_seg), dtype=np.float64)
    l = np.fromiter((len(a) for a in sub_seg), dtype=int)

    # construct union-find template
    fa = np.arange(len(start_idx))

    def find(i: int):
        if fa[i] != i:
            fa[i] = find(fa[i])
        return fa[i]

    def merge(i: int, j: int):
        pi, pj = find(i), find(j)
        if pi == pj:
            return
        u1, u2 = u[pi], u[pj]
        s1, s2 = s[pi], s[pj]
        l1, l2 = l[pi], l[pj]
        u[pj] = (l1 * u1 + l2 * u2) / (l1 + l2)
        s[pj] = (
            (l1 - 1) * s1
            + l1 * (u[pj] - u1) ** 2
            + (l2 - 1) * s2
            + l2 * (u[pj] - u2) ** 2
        ) / (l1 + l2 - 1)
        l[pj] = l1 + l2
        fa[pi] = pj

    def calc_pvalue(i: int, j: int):
        pi, pj = find(i), find(j)
        u1, u2 = u[pi], u[pj]
        s1, s2 = s[pi], s[pj]
        l1, l2 = l[pi], l[pj]
        (_, p) = stats.ttest_ind_from_stats(u1, np.sqrt(s1), l1, u2, np.sqrt(s2), l2)
        return p

    def link_neighbor():
        for i in range(1, len(start_idx)):
            pi, pj = find(i), find(i - 1)
            if pi != pj:
                continue
            start_idx[i] = start_idx[i - 1]
            start_idx[i - 1] = 0
        idx = start_idx != 0
        idx[0] = True
        return idx

    # merge similar neighbors
    idx = np.argsort(np.abs(np.diff(u)))
    for i in idx:
        pi, pj = find(i + 1), find(i)
        if pi != pj and calc_pvalue(pi, pj) > 0.05:
            merge(pi, pj)
    for i in range(len(fa)):
        p = find(i)
        u[i], s[i], l[i] = u[p], s[p], l[p]
    idx = link_neighbor()
    start_idx = start_idx[idx]
    u, s, l = u[idx], s[idx], l[idx]
    fa = np.arange(len(start_idx))

    # merge global similar segment
    flag = True
    while flag:
        flag = False
        idx = np.argsort(u)
        idx2 = np.argsort(np.abs(np.diff(np.sort(u))))
        for i in idx2:
            pi, pj = find(idx[i + 1]), find(idx[i])
            if pi != pj and calc_pvalue(pi, pj) > 0.05:
                merge(pi, pj)
                flag = True
        for i in range(len(fa)):
            p = find(i)
            u[i], s[i], l[i] = u[p], s[p], l[p]
    idx = link_neighbor()
    start_idx = start_idx[idx]
    u, s, l = u[idx], s[idx], l[idx]

    segment = np.c_[start_idx, np.concatenate((start_idx[1:], [len(data)]))]

    return segment, u


# 1-dimensional case quick computation
# (Rousseeuw, P. J. and Leroy, A. M. (2005) References, in Robust
#  Regression and Outlier Detection, John Wiley & Sons, chapter 4)
def estimate_location(data: np.ndarray) -> float:
    n_samples = len(data)
    n_support = int(n_samples * 0.6)
    X_sorted = np.sort(np.ravel(data))
    diff = X_sorted[n_support:] - X_sorted[: (n_samples - n_support)]
    halves_start = np.where(diff == np.min(diff))[0]
    # take the middle points' mean to get the robust location estimate
    location = (
        0.5 * (X_sorted[n_support + halves_start] + X_sorted[halves_start]).mean()
    )
    return location


def ostu_classifier(segment: np.ndarray, data: np.ndarray, normal: float):
    """Classify each segment

    Params
    ------
    segment: array, shape(N, 2), like [[start, end),...]
    data: float array, the value of each segment
    normal: float, normal value

    Returns
    -------
    segment: array, shape(N, 2), like [[start, end),...]
    state: array, length is N, value is 1->gain,0->normal,-1->loss
    """
    dist = np.abs(data - normal)
    total = len(dist)
    total_u = dist.mean()
    dist[dist > total_u] = total_u
    total_u = dist.mean()
    sorted_data = sorted(Counter(dist).items(), key=lambda x: x[0])
    ans, max_v = sorted_data[0][0], -1
    cur_w = sorted_data[0][1] / total
    cur_u = sorted_data[0][0] * cur_w

    for i in range(1, len(sorted_data)):
        cur_data = sorted_data[i][0]
        sigma = (total_u * cur_w - cur_u) ** 2 / (cur_w * (1 - cur_w))
        if sigma > max_v:
            max_v = sigma
            ans = cur_data
        tmp_w = sorted_data[i][1] / total
        cur_w += tmp_w
        cur_u += cur_data * tmp_w
        if (cur_data + normal) / normal * 2 - 2 > 2:
            break

    labels = dist >= ans
    state = np.select([labels & (data > normal), labels & (data < normal)], [1, -1], 0)
    for i in range(1, len(state)):
        if state[i] != state[i - 1]:
            continue
        segment[i, 0] = segment[i - 1, 0]
        segment[i - 1] = 0
    idx = segment.any(axis=1)
    segment, state = segment[idx], state[idx]

    return segment, state


def get_result(
    segment: np.ndarray, state: np.ndarray, pos: np.ndarray, bp_per_bin: int
):
    """Generate DataFrame CNV detetion result

    Params
    ------
    segment: array, shape(N, 2), like [[start, end),...]
    state: array, length is N, value is 1->gain,0->normal,-1->loss
    pos: array, the bp position to bin index
    bp_per_bin: int, the number of bp in each bin

    Returns
    -------
    result: pd.DataFrame, title is (start, end, type), type is (gain|loss)
    """
    idx = state != 0
    test_start = pos[segment[idx, 0]] * bp_per_bin + 1
    test_end = pos[segment[idx, 1] - 1] * bp_per_bin + bp_per_bin
    test_type = np.select([state == 1, state == -1], ["gain", "loss"], "")
    test_type = test_type[test_type != ""]
    result = pd.DataFrame(
        data=zip(test_start, test_end, test_type), columns=["start", "end", "type"]
    )
    return result


def OTSUCNV(data: RDdata, *, bp_per_bin: int = 1000) -> pd.DataFrame:
    """Copy number variations detection.

    Parameters
    ----------
    data: RDdata
    Generated by `binning` function

    bp_per_bin: int
    The length of each bin

    Returns
    -------
    result: pd.DataFrame, title is (`start`, `end`, `type`), and the `type` contains (gain|loss)
    """
    _, start_idx = init_segment(data.RD)
    normal = estimate_location(data.RD)
    segment, u = rearrange_segment_value(data.RD, start_idx)
    segment, state = ostu_classifier(segment, u, normal)
    result = get_result(segment, state, data.pos, bp_per_bin)
    return result
