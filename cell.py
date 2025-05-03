from globals import *

class Cell:
    def __init__(self, cell_index, nodes, height, neuron=False):
        """
        Hexagonal cell with 6 nodes, a height value, and a neuron flag.
        """
        if len(nodes) != 6:
            raise ValueError("A Cell must have exactly 6 nodes.")
        self.cell_index = cell_index
        self.nodes = nodes
        self.height = height
        self.neuron = neuron

    def __repr__(self):
        return f"Cell(index={self.cell_index}, height={self.height:.2f}, neuron={self.neuron})"
    
    def is_neuron(self):
        """
        Returns True if the cell is a neuron, False otherwise.
        """
        return self.neuron
    
    def get_height(self):
        """
        Returns the height of the cell.
        """
        return self.height
    
    def get_nodes(self):
        """
        Returns the nodes of the cell.
        """
        return self.nodes

    def update_height(self, new_height):
        """
        Updates the height of the cell.
        """
        self.height = new_height

