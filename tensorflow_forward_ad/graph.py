from __future__ import (division, print_function, unicode_literals)

import numpy as np

from tensorflow_forward_ad import logger
from tensorflow_forward_ad.cbfs import bfs

log = logger.get()


def get_path_cover(adj, src, dst, bfs_cache=None):
  """Generates a list of nodes that contains all the nodes in between the
  source and the destination (endpoints included).

  Args:
    adj: Adjacency matrix.
    src: Source node index, 0-based.
    dst: Destination node index, 0-based.

  Returns:
    cover: List of node indices, in traversal order.
  """
  return get_path_cover_multi_src(adj, [src], dst, bfs_cache=bfs_cache)


def get_path_cover_multi_src(adj, src_list, dst, bfs_cache=None):
  """Generates a list of nodes that contains all the nodes in between
  multiple sources and a destination (endpoints included).

  Args:
    adj: Adjacency matrix.
    src: Source node index list, 0-based.
    dst: Destination node index, 0-based.

  Returns:
    cover: List of node indices, in traversal order.
  """
  num_nodes = adj.shape[0]
  if bfs_cache is None:
    bfs_cache = np.zeros([num_nodes, num_nodes], dtype=np.int8)
  cover = []
  for src in src_list:
    if src == dst:
      return cover
    if src > dst:
      raise Exception("Source must be smaller than destination.")
    if src >= num_nodes or dst >= num_nodes:
      raise Exception("Node index must be smaller than number of nodes.")
    if not bfs(adj, src, dst, cache=bfs_cache):
      log.warning("Source is not connected to destination.")
  cover.extend(src_list)
  start = min(src_list) + 1
  for idx in range(start, dst):
    # If the node is connected to both source and destination, add.
    if idx in src_list:
      continue
    connect_dst = bfs(adj, idx, dst, cache=bfs_cache)
    if not connect_dst:
      continue
    for src in src_list:
      connect_src = bfs(adj, src, idx, cache=bfs_cache)
      if connect_src:
        break
    if connect_src and connect_dst:
      cover.append(idx)
  cover.append(dst)
  return cover


def format_tensor_name(name):
  """Strips the tensor name to reflect the op name."""

  if name.startswith("^"):
    name_old = name
    name = name.strip("^")
    log.warning("Changing \"{}\" to \"{}\"".format(name_old, name))
  return name.split(":")[0]
  # return name


def convert_node_list_to_adj_mat(node_list):
  """Converts a node list into an adjacency matrix.

  Args:
    node_list: List of NodeDef objects.

  Returns:
    adj_mat: Adjacency matrix, np.array.
    ord_table: Mapping from node name to index, in a graph traversal order.
  """
  idx_table = {}
  ord_table = {}

  def add_node(table, node):
    if node.name in table:
      return
    for subnode_name in node.input:
      # Not sure if this will fix the variable
      # assign issue.
      # sbn = subnode_name.strip("^")\
      sbn = subnode_name
      sbn = format_tensor_name(sbn)
      if sbn not in table:
        if sbn not in idx_table:
          log.fatal("Not found {}".format(sbn))
          continue
        add_node(table, idx_table[sbn])
    table[node.name] = len(table)

  for node in node_list:
    idx_table[node.name] = node

  for node in node_list:
    add_node(ord_table, node)

  NN = len(node_list)
  adj_mat = [0] * (NN**2)
  for node in node_list:
    dst = ord_table[node.name]
    for inp in node.input:
      inp_ = format_tensor_name(inp)
      src = ord_table[inp_]
      adj_mat[src * NN + dst] = 1
  adj_mat = np.array(adj_mat, dtype=np.int8).reshape([NN, NN])
  return adj_mat, ord_table


def get_path_cover_str(node_list, src, dst):
  """Gets the path cover in string format.

  Args:
    node_list: List of NodeDef objects.
    src: Source node name, string.
    dst: Destination node name, string.

  Returns:
    list of node defs to traverse.
  """
  return get_path_cover_str_list(node_list, src, [dst])


def get_path_cover_str_list(node_list, src, dst_list):
  """Gets the path cover in string format.

  Args:
    node_list: List of NodeDef objects.
    src: Source node name, string.
    dst: Destination node name, list of string.

  Returns:
    list of node defs to traverse.
  """
  return get_path_cover_str_list_list(node_list, [src], dst_list)


def get_path_cover_str_list_list(node_list, src_list, dst_list):
  """Gets the path cover in string format.

  Args:
    node_list: List of NodeDef objects.
    src_list: Source node name, string, list of string.
    dst_list: Destination node name, list of string.

  Returns:
    list of node defs to traverse.
  """
  adj_mat, ord_table = convert_node_list_to_adj_mat(node_list)
  src_idx_list = [ord_table[format_tensor_name(src)] for src in src_list]
  dst_idx_list = [ord_table[format_tensor_name(dst)] for dst in dst_list]
  added_nodes = set()
  # Build inverse table from integer => string.
  ord_table_inv = [None] * len(node_list)
  num_nodes = len(node_list)
  bfs_cache = np.zeros([num_nodes, num_nodes], dtype=np.int8)
  for node, idx in ord_table.items():
    ord_table_inv[idx] = node
  path_cover_str = []
  # log.info("Computation graph BFS traversal...")
  # for dst_idx in tqdm(dst_idx_list):
  for dst_idx in dst_idx_list:
    path_cover = get_path_cover_multi_src(
        adj_mat, src_idx_list, dst_idx, bfs_cache=bfs_cache)
    for node_idx in path_cover:
      if node_idx not in added_nodes:
        path_cover_str.append(ord_table_inv[node_idx])
        added_nodes.add(node_idx)
  return path_cover_str
