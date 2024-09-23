import sys, os
import json
import numpy as np
import pandas as pd
import urllib
from collections import OrderedDict, Counter
import operator
import networkx as nx
from robustness.datasets import DATASETS

REQUIRED_FILES = ['dataset_class_info.json',
                  'class_hierarchy.txt',
                  'node_names.txt']

def setup_breeds(info_dir, url=BREEDS_URL):
    print(f"Downloading files from {url} to {info_dir}")
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    for f in REQUIRED_FILES:
        urllib.request.urlretrieve(f"{url}/{f}?raw=true",
                                   os.path.join(info_dir, f))

class ClassHierarchy():
    '''
    Class representing a general ImageNet-style hierarchy.
    '''
    def __init__(self, info_dir, root_wnid='n00001740'):
        """
        Args:
            info_dir (str) : Path to hierarchy information files. Contains a 
                "class_hierarchy.txt" file with one edge per line, a
                "node_names.txt" mapping nodes to names, and "class_info.json".
        """

        for f in REQUIRED_FILES:
            if not os.path.exists(os.path.join(info_dir, f)):
                raise Exception('Missing files: `info_dir` does not contain required file {f}')

        # Details about dataset class names (leaves), IDS
        with open(os.path.join(info_dir, "dataset_class_info.json")) as f:
            class_info = json.load(f)

        # Hierarchy represented as edges between parent & child nodes.
        with open(os.path.join(info_dir, "class_hierarchy.txt")) as f:
            edges = [l.strip().split() for l in f.readlines()]

        # Information (names, IDs) for intermediate nodes in hierarchy.
        with open(os.path.join(info_dir, "node_names.txt")) as f:
            mapping = [l.strip().split('\t') for l in f.readlines()]


        # Original dataset classes
        # leaf_nodes = [c[1] for c in class_info]
        self.CLASS_IDX = [c[1] for c in class_info]
        self.LEAF_ID_TO_NAME = {c[1]: c[2] for c in class_info}
        self.LEAF_ID_TO_NUM = {c[1]: c[0] for c in class_info}
        self.LEAF_NUM_TO_NAME = {c[0]: c[2] for c in class_info}   

        # Full hierarchy
        self.HIER_NODE_NAME = {w[0]: w[1] for w in mapping}
        self.graph = self._make_parent_graph(self.CLASS_IDX, edges)
        self.CLASS_IDX = [9999 for _ in mapping]

        # Arrange nodes in levels (top-down)
        self.node_to_level = self._make_level_dict(self.graph, root=root_wnid)
        self.level_to_nodes = {}
        for k, v in self.node_to_level.items():
            if v not in self.level_to_nodes: self.level_to_nodes[v] = []
            self.level_to_nodes[v].append(k)

    @staticmethod
    def _make_parent_graph(nodes, edges):
        """
        Obtain networkx representation of class hierarchy.

        Args:
            nodes [str] : List of node names to traverse upwards.
            edges [(str, str)] : Tuples of parent-child pairs.

        Return:
            networkx representation of the graph.
        """

        # create full graph
        full_graph_dir = {}
        for p, c in edges:
            if p not in full_graph_dir:
                full_graph_dir[p] = {c: 1}
            else:
                full_graph_dir[p].update({c: 1})
                    
        FG = nx.DiGraph(full_graph_dir)

        # perform backward BFS to get the relevant graph
        graph_dir = {}
        todo = [n for n in nodes if n in FG.nodes()] # skip nodes not in graph
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for p in FG.predecessors(w):
                    if p not in graph_dir:
                        graph_dir[p] = {w: 1}
                    else:
                        graph_dir[p].update({w: 1})
                    todo.append(p)
            todo = set(todo)
        
        return nx.DiGraph(graph_dir)

    @staticmethod
    def _make_level_dict(graph, root):
        """
        Map nodes to their level within the hierarchy (top-down).

        Args:
            graph (networkx graph( : Graph representation of the hierarchy
            root (str) : Hierarchy root.

        Return:
            Dictionary mapping node names to integer level.
        """    

        level_dict = {} 
        todo = [(root, 0)] # (node, depth)
        while todo:
            curr = todo
            todo = []
            for n, d in curr:
                if n not in level_dict:
                    level_dict[n] = d
                else:
                    level_dict[n] = max(d, level_dict[n]) # keep longest path
                for c in graph.successors(n):
                    todo.append((c, d + 1))

        return level_dict

    def leaves_reachable(self, n):
        """
        Determine the leaves (ImageNet classes) reachable for a give node.

        Args:
            n (str) : WordNet ID of node

        Returns:
            leaves (list): List of WordNet IDs of the ImageNet descendants
        """    
        leaves = set()
        todo = [n]
        while todo:
            curr = todo
            todo = []
            for w in curr:
                for c in self.graph.successors(w):
                    if c in self.CLASS_IDX:
                        leaves.add(c)
                    else:
                        todo.append(c)
            todo = set(todo)

        # If the node itself is an ImageNet node
        if n in self.CLASS_IDX: leaves = leaves.union([n])
        return leaves

    def node_name(self, n):
        """
        Determine the name of a node.
        """    
        if n in self.HIER_NODE_NAME:
            return self.HIER_NODE_NAME[n]
        else:
            return n

    def print_node_info(self, nodes):
        """
        Prints basic information (name, number of ImageNet descendants) 
        about a given set of nodes.

        Args:
            nodes (list) : List of WordNet IDs for relevant nodes
        """    

        for n in nodes:
            if n in self.HIER_NODE_NAME:
                print_str = f"{n}: {self.HIER_NODE_NAME[n]}"
                print_str += f"\n indexx {n}: {self.CLASS_IDX[n]}"
            else:
                print_str = n

            print_str += f" ({len(self.leaves_reachable(n))})"
            print(print_str)

    def traverse(self, nodes, direction='down', depth=100):
        """
        Find all nodes accessible from a set of given nodes.

        Args:
            nodes (list) : List of WordNet IDs for relevant nodes
            direction ("up"/"down"): Traversal direction
            depth (int): Maximum depth to traverse (from nodes)

        Returns:
            Set of nodes reachable within the desired depth, in the
            desired direction.
        """    

        if not nodes or depth == 0:
            return nodes

        todo = []
        for n in nodes:
            if direction == 'down':
                todo.extend(self.graph.successors(n))
            else: 
                todo.extend(self.graph.predecessors(n))
        return nodes + self.traverse(todo, direction=direction, depth=depth-1)

    def get_nodes_at_level(self, L, ancestor=None):
        """
        Find all superclasses at a specified depth within a subtree
        of the hierarchy.

        Args:
            L (int): Depth in hierarchy (from root node)
            ancestor (str): (optional) WordNet ID that can be used to
                            restrict the subtree in which superclasses
                            are found

        Returns:
            nodes (list): Set of superclasses at that depth in 
                                   the hierarchy
        """            
        if ancestor is not None:
            valid = set(self.traverse([ancestor], direction="down"))

        nodes = set([v for v in self.level_to_nodes[L]
                     if ancestor is None or v in valid])
        return nodes