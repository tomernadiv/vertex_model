def half_tissue(row: int, num_rows: int):
    return row > (int(num_rows / 2) - 1)

def all_neurons(row: int, num_rows: int):
    return True

def all_non_neurons(row: int, num_rows: int):
    return False

def outline(num_layers: int, num_frames: int, row: int, num_rows: int, col: int, num_cols: int) -> bool:

    # Determine which frame this column belongs to
    frame_start, frame_end, frame_index = get_frame_borders(col, num_cols, num_frames)

    # Check row and column against the outline within the frame
    return (
        row < num_layers or row >= (num_rows - num_layers) or
        col < (frame_start + num_layers) or col >= (frame_end - num_layers)
    )

# --- Frame utilities ---

def get_frame_bounds_from_index(frame_index: int, num_cols: int, num_frames: int):
    frame_width = num_cols // num_frames
    frame_start = frame_index * frame_width
    frame_end = (frame_index + 1) * frame_width if frame_index < num_frames - 1 else num_cols
    return frame_start, frame_end

def get_frame_borders(col: int, num_cols: int, num_frames: int):
    frame_width = num_cols // num_frames
    frame_index = min(col // frame_width, num_frames - 1)
    frame_start, frame_end = get_frame_bounds_from_index(frame_index, num_cols, num_frames)
    return frame_start, frame_end, frame_index

# --- Inner outline logic ---

def get_inner_frame_bounds_from_index(frame_index: int, num_cols: int, num_rows: int, num_frames: int, num_layers: int):
    frame_start, frame_end = get_frame_bounds_from_index(frame_index, num_cols, num_frames)

    inner_left = frame_start + num_layers - 1
    inner_right = frame_end - num_layers
    inner_top = num_layers - 1
    inner_bottom = num_rows - num_layers

    return inner_left, inner_right, inner_top, inner_bottom

def get_inner_frame_bounds_for_col(col: int, num_cols: int, num_rows: int, num_frames: int, num_layers: int):
    _, _, frame_index = get_frame_borders(col, num_cols, num_frames)
    return get_inner_frame_bounds_from_index(frame_index, num_cols, num_rows, num_frames, num_layers)

def inner_outline(num_layers: int, num_frames: int, row: int, num_rows: int, col: int, num_cols: int,
                  inner_border_thickness: int = 1) -> tuple[bool, str | None]:

    inner_left, inner_right, inner_top, inner_bottom = get_inner_frame_bounds_for_col(
        col, num_cols, num_rows, num_frames, num_layers
    )

    if not (inner_top <= row < inner_bottom and inner_left <= col < inner_right):
        return (False, None)

    if inner_top <= row < inner_top + inner_border_thickness:
        return (True, "top")
    if inner_bottom - inner_border_thickness <= row < inner_bottom:
        return (True, "bottom")
    if inner_left <= col < inner_left + inner_border_thickness:
        return (True, "left")
    if inner_right - inner_border_thickness <= col < inner_right:
        return (True, "right")

    return (False, None)

# --- Frame bounds generator ---

def get_all_frame_bounds(num_frames: int, num_cols: int, num_rows: int, num_layers: int = 0):
    frame_bounds = []
    for frame_index in range(num_frames):
        x_start, x_end = get_frame_bounds_from_index(frame_index, num_cols, num_frames)
        inner_left = x_start + num_layers - 1
        inner_right = x_end - num_layers
        inner_top = num_layers - 1
        inner_bottom = num_rows - num_layers
        frame_bounds.append((inner_left, inner_right, inner_top, inner_bottom))
    return frame_bounds




# """
#     function with the logic of how to initiate the neurons in the tissue. 
# """
# def half_tissue(row:int, num_rows:int):
#     return row > (int(num_rows/2) - 1)

# def all_neurons(row:int, num_rows:int):
#     return True

# def all_non_neurons(row:int, num_rows:int):
#     return False


# def get_frame_borders(col, num_cols, num_frames):
#     # Calculate frame width
#     frame_width = num_cols // num_frames

#     # Determine which frame this column belongs to
#     frame_index = min(col // frame_width, num_frames - 1)
#     frame_start = frame_index * frame_width
#     frame_end = (frame_index + 1) * frame_width if frame_index < (num_frames - 1) else num_cols

#     return frame_start, frame_end, frame_index

# def get_inner_frame_bounds_for_col(col: int, num_frames: int, num_cols: int, num_rows: int, num_layers: int):
#     frame_start, frame_end, _ = get_frame_borders(col, num_cols, num_frames)

#     inner_top = num_layers - 1
#     inner_bottom = num_rows - num_layers
#     inner_left = frame_start + num_layers - 1
#     inner_right = frame_end - num_layers

#     return inner_left, inner_right, inner_top, inner_bottom

# def inner_outline(num_layers: int,num_frames: int,row: int,num_rows: int,col: int,num_cols: int,
# inner_border_thickness: int = 1
# ) ->  tuple[bool, str | None]:
        
#     inner_top, inner_bottom, inner_left, inner_right = get_inner_frame_bounds_for_col(col, num_frames, num_cols, num_rows, num_layers)

#     # Check if we're even within the inner area bounds
#     if not (inner_top <= row < inner_bottom and inner_left <= col < inner_right):
#         return (False, None)

#     # Top edge: row is within top thickness, and col is within the frame horizontally
#     if inner_top <= row < inner_top + inner_border_thickness:
#         return (True, "top")

#     # Bottom edge: row is within bottom thickness, and col is within the frame horizontally
#     if inner_bottom - inner_border_thickness <= row < inner_bottom:
#         return (True, "bottom")

#     # Left edge: col is within left thickness, and row is within the frame vertically
#     if inner_left <= col < inner_left + inner_border_thickness:
#         return (True, "left")

#     # Right edge: col is within right thickness, and row is within the frame vertically
#     if inner_right - inner_border_thickness <= col < inner_right:
#         return (True, "right")


#     return (False, None)



# def get_all_frame_bounds(num_frames: int, num_cols: int, num_rows: int, num_layers: int = 0):
#     """
#     Return a list of (x_start, x_end, y_start, y_end) for all frames.
#     Optionally trims `num_layers` from each side of the frame.
#     """
#     frame_width = num_cols // num_frames
#     frame_bounds = []

#     for frame_index in range(num_frames):
#         x_start = frame_index * frame_width + num_layers
#         x_end = (frame_index + 1) * frame_width if frame_index < num_frames - 1 else num_cols
#         x_end -= num_layers

#         y_start = num_layers
#         y_end = num_rows - num_layers

#         frame_bounds.append((x_start, x_end, y_start, y_end))

#     return frame_bounds

