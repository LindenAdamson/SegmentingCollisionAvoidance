import numpy as np
from numba import njit, prange

def compile_numba_functs():
    compile_make_cartesian_njit_helper()
    compile_find_ground_njit_helper()

def compile_make_cartesian_njit_helper():
    rows = cols = 2
    lr = ud = np.zeros(2)
    depthImg = np.zeros((2,2))
    cartImg = np.zeros((2,2,3))
    make_cartesian_njit_helper(rows, cols, lr, ud, depthImg, cartImg)

@njit(parallel=True)
def make_cartesian_njit_helper(rows, cols, lr, ud, depthImg, cartImg):
    for i in prange(rows):
        for j in prange(cols):
            rho = depthImg[i,j]
            x = rho * lr[j]
            y = rho * ud[i]
            z = rho
            cartImg[i,j,0] = x
            cartImg[i,j,1] = -y
            cartImg[i,j,2] = z

def compile_find_ground_njit_helper():
    rows = cols = 2
    ground = np.zeros((2,2))
    outline_matrix = buckets_matrix = np.zeros((2,2,3))
    find_ground_njit_helper(rows, cols, ground, outline_matrix, buckets_matrix)

@njit(parallel=True)
def find_ground_njit_helper(rows, cols, ground, outline_matrix, buckets_matrix):
    for column in prange(cols):
        outline = outline_matrix[column,:,:]
        start = np.where(np.min(outline[:,0]) == outline[:,0])[0][0]
        lda = .25
        max_dist = 1
        max_angle = .5
        curr_bucket = np.max(outline[:,0]) ** 2
        prev_bucket = np.max(outline[:,0]) ** 2
        buckets = buckets_matrix[column,:,:]
        bucket_i = -1
        for p_i in range(start, rows):
            point = outline[p_i,0:2]
            if abs(point[0] - curr_bucket) < lda:
                outline[p_i,2] = bucket_i
                buckets[bucket_i, 0:2] += point
                buckets[bucket_i, 2] += 1
                curr_bucket = buckets[bucket_i, 0] / buckets[bucket_i, 2]
                continue
            if abs(point[0] - prev_bucket) < lda:
                outline[p_i,2] = bucket_i - 1
                buckets[bucket_i - 1, 0:2] += point
                buckets[bucket_i - 1, 2] += 1
                prev_bucket = buckets[bucket_i - 1, 0] / buckets[bucket_i - 1, 2]
                continue
            bucket_i += 1
            prev_bucket = curr_bucket
            curr_bucket = point[0]
            buckets[bucket_i, 2] += 1
            buckets[bucket_i, 0:2] += point
        # buckets = buckets[:bucket_i + 1,0:2] / buckets[:bucket_i + 1,2].reshape(bucket_i + 1,1)
        resized_buckets = buckets[:bucket_i + 1,0:2]
        resized_buckets[:,0] /= buckets[:bucket_i + 1,2]
        resized_buckets[:,1] /= buckets[:bucket_i + 1,2]
        dist = np.sum(np.square(resized_buckets[:-1,:] - resized_buckets[1:,:]), axis=1)
        angle = np.acos(np.abs(resized_buckets[:-1,0] - resized_buckets[1:,0]) ** 2 / dist)
        done = resized_buckets.shape[0] - 1
        for i in range(resized_buckets.shape[0] - 2):
            if dist[i] > max_dist or abs(angle[i] - angle[i + 1]) > max_angle:
                done = i + 1
                break
        if done >= resized_buckets.shape[0]:
            done = resized_buckets.shape[0] - 1
        c = resized_buckets[done]
        mini = np.max(outline[:,0]) - np.min(outline[:,0])
        for i in range(start, rows):
            dist = np.sum(np.square(c - outline[i,0:2]))
            if dist < mini:
                mini = dist
                idx = i
        ground[rows - idx:,column] = True