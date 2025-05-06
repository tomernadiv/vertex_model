"""
    function with the logic of how to initiate the neurons in the tissue. 
"""
def half_tissue(row:int, num_rows:int):
    return row > (int(num_rows/2) - 1)

def all_neurons(row:int, num_rows:int):
    return True

def all_non_neurons(row:int, num_rows:int):
    return False

def outline(num_layers:int, row:int, num_rows:int, col:int, num_cols:int):
        return (
        row < num_layers or row >= (num_rows - num_layers) or
        col < num_layers or col >= (num_cols - num_layers)
    )




