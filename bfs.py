"""
Implements basic graph algorithms such as BFS.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
"""

import numpy as np


def bfs(adj, src, dst, cache=None):
  """BFS search from source to destination. Check whether a path exists, does
  not return the actual path.
  Work on directed acyclic graphs where we assume that there is no path to the
  node itself.

  Args:
    adj: Adjacency matrix.
    src: Source node index, 0-based.
    dst: Destination node index, 0-based.
    cache: 2D matrix, cache[i, j] = 1 indicates path exists between two node
      i and j. cache[i, j] = -1 indicates path does not exists between two node
      i and j. chace[i, j] = 0 indicates unknown.

  Returns:
    found: A path is found between source and destination.
  """
  if src == dst: return False
  num_nodes = adj.shape[0]
  if num_nodes == 0:
    return False
  if src >= num_nodes or dst >= num_nodes:
    raise Exception("Index must be smaller than the number of nodes.")
  if num_nodes == 1:
    return False
  # Whether a node has been visited, if not negative, the parent.
  parent = np.zeros([num_nodes], dtype=np.int64) - 1
  nodes_to_visit = [src]
  found = False
  # BFS loop.
  while len(nodes_to_visit) > 0:
    cur = nodes_to_visit.pop(0)
    if cur == dst:
      found = True
      break
    if cache is not None:
      if cache[cur, dst] == 1:
        found = True
        break
      elif cache[cur, dst] == -1:
        continue

    for jj in range(num_nodes):
      if adj[cur, jj] == 1:
        if parent[jj] == -1:
          nodes_to_visit.append(jj)
          parent[jj] = cur

  if not found:
    # Add the source node to the cache.
    if cache is not None:
      #log.info(("Setting -1", src, dst), verbose=2)
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
        #log.info(("Setting 1", cur, dst), verbose=2)
      cache[src, dst] = 1
      #log.info(("Setting 1", src, dst), verbose=2)
    return True
