"""
    function with the logic of how to initiate the neurons in the tissue. 
"""
def half_tissue(row:int, num_rows:int):
    return row > (int(num_rows/2) - 1)

def all_neurons(row:int, num_rows:int):
    return True

def all_non_neurons(row:int, num_rows:int):
    return False

def outline(num_layers: int, num_frames: int, row: int, num_rows: int, col: int, num_cols: int) -> bool:
    # Calculate frame width
    frame_width = num_cols // num_frames

    # Determine which frame this column belongs to
    frame_index = min(col // frame_width, num_frames - 1)  # avoid overflow on last column
    frame_start = frame_index * frame_width
    frame_end = (frame_index + 1) * frame_width if frame_index < (num_frames - 1) else num_cols

    # Check row and column against the outline within the frame
    return (
        row < num_layers or row >= (num_rows - num_layers) or
        col < (frame_start + num_layers) or col >= (frame_end - num_layers)
    )

def get_frame_borders(col, num_cols, num_frames):
    # Calculate frame width
    frame_width = num_cols // num_frames

    # Determine which frame this column belongs to
    frame_index = min(col // frame_width, num_frames - 1)
    frame_start = frame_index * frame_width
    frame_end = (frame_index + 1) * frame_width if frame_index < (num_frames - 1) else num_cols

    return frame_start, frame_end

def inner_outline(num_layers: int,num_frames: int,row: int,num_rows: int,col: int,num_cols: int,
inner_border_thickness: int = 1
) -> bool:
    
    frame_start, frame_end  = get_frame_borders(col, num_cols, num_frames)

    # Define inner window bounds
    inner_top = num_layers
    inner_bottom = num_rows - num_layers
    inner_left = frame_start + num_layers
    inner_right = frame_end - num_layers

    # Check if we're even within the inner area bounds
    if not (inner_top <= row < inner_bottom and inner_left <= col < inner_right):
        return False

    # Top edge: row is within top thickness, and col is within the frame horizontally
    top_edge = inner_top <= row < inner_top + inner_border_thickness

    # Bottom edge: row is within bottom thickness, and col is within the frame horizontally
    bottom_edge = inner_bottom - inner_border_thickness < row <= inner_bottom

    # Left edge: col is within left thickness, and row is within the frame vertically
    left_edge = inner_left <= col < inner_left + inner_border_thickness

    # Right edge: col is within right thickness, and row is within the frame vertically
    right_edge = inner_right - inner_border_thickness < col <= inner_right


    return top_edge or bottom_edge or left_edge or right_edge







