"""
Basic tests of BFS.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
import numpy as np
import unittest

from tensorflow_forward_ad.cbfs import bfs


class BFSTests(unittest.TestCase):

  def test_bfs(self):
    adj = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
    self.assertTrue(bfs(adj, 0, 2, None))
    self.assertFalse(bfs(adj, 2, 0, None))

  def test_bfs_cache(self):
    adj = np.zeros([3, 3], dtype=np.int8)
    self.assertTrue(bfs(adj, 0, 2, cache=np.ones([3, 3], dtype=np.int8)))
    self.assertFalse(bfs(adj, 0, 2, cache=np.zeros([3, 3], dtype=np.int8)))

  def test_bfs_performance(self):
    adj = np.random.uniform(0, 1, [1000, 1000])
    adj = (adj > 0.3).astype(np.int8)
    for ii in range(10):
      bfs(adj, 0, 999, None)

  pass


if __name__ == "__main__":
  unittest.main()
