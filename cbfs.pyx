"""
BFS Cython implementation.
Author: Mengye Ren (mren@cs.toronto.edu)
"""
import cython
import numpy as np
cimport numpy as np

DEF MAX_NUM_NODES = 10000

@cython.boundscheck(False)
@cython.wraparound(False)
def bfs(np.ndarray[np.int8_t, ndim=2, mode="c"] adj, int src, int dst, np.ndarray[np.int8_t, ndim=2, mode="c"] cache):
  if src == dst: return False
  cdef int num_nodes = adj.shape[0]
  assert num_nodes <= MAX_NUM_NODES, "Exceeds maximum number of nodes."
  assert src < num_nodes and dst < num_nodes, "Index must be smaller than the number of nodes."
  if num_nodes == 0:
    return False
  if num_nodes == 1:
    return False
  # Whether a node has been visited, if not negative, the parent.
  cdef int parent[MAX_NUM_NODES]
  # A queue storing the nodes to be visited.
  cdef int nodes_to_visit[MAX_NUM_NODES]
  for ii in range(num_nodes):
    parent[ii] = -1
    nodes_to_visit[ii] = 0
  nodes_to_visit[0] = src
  cdef int qlen = 1
  cdef int qend = 1
  cdef int qstart = 0
  cdef int found = 0

  # BFS loop.
  while qlen > 0:
    cur = nodes_to_visit[qstart]
    qlen -= 1
    qstart += 1
    if qstart == num_nodes:
      qstart = 0
    if cur == dst:
      found = 1
      break
    if cache is not None:
      if cache[cur, dst] == 1:
        found = 1
        break
      elif cache[cur, dst] == -1:
        continue

    for jj in range(num_nodes):
      if adj[cur, jj] == 1:
        if parent[jj] == -1:
          nodes_to_visit[qend] = jj
          parent[jj] = cur
          qlen += 1
          qend += 1
          if qend == num_nodes:
            qend = 0

  if found == 0:
    # Add the source node to the cache.
    if cache is not None:
      for ii in range(num_nodes):
        if parent[ii] >= 0:
          cache[ii, dst] = -1
      cache[src, dst] = -1
    return False
  else:
    # Add all the nodes to the cache.
    if cache is not None:
      # Backtrack.
      while cur != src:
        cur = parent[cur]
        cache[cur, dst] = 1
      cache[src, dst] = 1
    return True
