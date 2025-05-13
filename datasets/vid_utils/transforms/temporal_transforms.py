import random
import warnings

import numpy as np

class TemporalCenterStrideCrop:
    """Temporally crop the given frame indices from the center with a stride.

    Selects `size` frames from the center of the sequence, with an interval
    of `stride` between them.

    If the effective length needed for the crop ((size - 1) * stride + 1)
    is less than the total number of frames, the sequence of frames
    selected starts from a centered position.

    If the total number of frames is less than the effective length needed,
    padding is applied if `padding` is True. The padding method defaults
    to 'loop', repeating the sequence until it's long enough.

    Args:
        size (int): Desired number of frames in the output crop.
        stride (int): The step/interval between selected frames.
        padding (bool): Whether to pad the frame indices if the total number
                        of frames is less than the length required for the
                        strided crop. Defaults to True.
        pad_method (str): Padding method. Currently, only 'loop' is implemented.
                        Defaults to 'loop'.
    """
    def __init__(self, size, stride, padding=True, pad_method='loop'):
        if size <= 0:
             raise ValueError("size must be positive")
        if stride <= 0:
             raise ValueError("stride must be positive")

        self.size = size
        self.stride = stride
        self.padding = padding
        self.pad_method = pad_method

        # Calculate the minimum number of frames required in the input
        # to get 'size' frames with 'stride' without padding
        self.required_length = (self.size - 1) * self.stride + 1

        if self.padding and self.pad_method != 'loop':
             # Note: The implementation below only provides 'loop' padding logic.
             warnings.warn(f"pad_method '{self.pad_method}' is specified, but only 'loop' padding is currently implemented.")
             exit(1)
    def _loop_pad(self, frame_indices, target_size):
        """Helper function to loop pad a list of indices."""
        if not frame_indices:
             # Cannot pad an empty list
             warnings.warn("Attempted to loop pad an empty list of frame indices.")
             return []

        out = list(frame_indices)
        # Calculate how many times we need to repeat the original sequence approximately
        repeat_factor = (target_size // len(out)) + 1 # +1 to be safe
        out = out * repeat_factor
        # Trim to the exact target size
        return out[:target_size]

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped and strided frame indices.
        """
        # Ensure input is a list for consistent operations
        frame_indices = list(frame_indices)
        n_frames = len(frame_indices)

        current_indices = frame_indices

        # 1. Handle insufficient frames and padding
        if n_frames < self.required_length:
            if self.padding:
                # Pad the indices to meet the required length for the strided crop
                if self.pad_method == 'loop':
                    current_indices = self._loop_pad(frame_indices, self.required_length)
                else:
                     # Fallback for unsupported padding methods, use loop as per warning
                     current_indices = self._loop_pad(frame_indices, self.required_length)
                # Update n_frames to the padded length
                n_frames = len(current_indices)
            else:
                # If not padding and not enough frames, we cannot fulfill the request
                raise ValueError(
                    f"Insufficient frames ({n_frames}) for the desired "
                    f"center stride crop (requires at least {self.required_length} "
                    f"frames for size={self.size}, stride={self.stride}) and padding is disabled."
                )

        # 2. Perform the center stride selection
        # Now, n_frames (length of current_indices) is guaranteed to be >= self.required_length
        current_n_frames = len(current_indices)

        # Calculate the range of possible start indices for a contiguous block of required_length
        # The earliest possible start index is 0
        # The latest possible start index is current_n_frames - self.required_length
        # The center of this range is (0 + (current_n_frames - self.required_length)) // 2
        start_index_for_block = (current_n_frames - self.required_length) // 2

        # The indices to select are start_index_for_block + i * self.stride for i from 0 to size-1
        out_indices = [current_indices[start_index_for_block + i * self.stride]
                       for i in range(self.size)]

        return out_indices
    
class LoopPadding:
    def __init__(self, size):
        self.size = size
    def __call__(self, frame_indices):
        out = list(frame_indices)
        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)
        return out

class TemporalCenterCrop:
    """Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size, padding=True, pad_method='loop'):
        self.size = size
        self.padding = padding
        self.pad_method = pad_method
    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))
        out = list(frame_indices[begin_index:end_index])
        if self.padding is True:
            if self.pad_method == 'loop':
                while len(out) < self.size:
                    for index in out:
                        if len(out) >= self.size:
                            break
                        out.append(index)
            else:
                while len(out) < self.size:
                    for index in out:
                        if len(out) >= self.size:
                            break
                        out.append(index)
                out.sort()
        return out

class TemporalRandomCrop:
    """Temporally crop the given frame indices at a random location.
        If the number of frames is less than the size,
        loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size=4, stride=8):
        self.size = size
        self.stride = stride
    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)
        if len(frame_indices) >= self.size * self.stride:
            rand_end = len(frame_indices) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.size:
            index = np.random.choice(len(frame_indices), size=self.size, replace=False)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        return out

class TemporalBeginCrop:
    """Temporally crop the given frame indices at a beginning.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size=4):
        self.size = size
    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        if len(frame_indices) >= 25:
            out = frame_indices[0:25:8]
        elif len(frame_indices) >= 13:
            out = frame_indices[0:13:4]
        elif len(frame_indices) >= 7:
            out = frame_indices[0:7:2]
        elif len(frame_indices) >= 4:
            out = frame_indices[0:4:1]
        else:
            out = frame_indices[0:4]
            while len(out) < 4:
                for index in out:
                    if len(out) >= 4:
                        break
                    out.append(index)
        return out

# class TemporalBeginCrop:
#     """Temporally crop the given frame indices at a beginning.

#     If the number of frames is less than the size,
#     loop the indices as many times as necessary to satisfy the size.

#     Args:
#         size (int): Desired output size of the crop.
#     """

#     def __init__(self, size=4):
#         self.size = size
        
#     def __call__(self, frame_indices):
#         frame_indices = list(frame_indices)

#         if len(frame_indices) >= 4:
#             out = frame_indices[0:4:1]
#         else:
#             out = frame_indices[0:4]
#             while len(out) < 4:
#                 for index in out:
#                     if len(out) >= 4:
#                         break
#                     out.append(index)

#         return out
if __name__ == '__main__':
    # Example 1: Sufficient frames
    indices1 = list(range(0, 100)) # 100 frames
    transform1 = TemporalCenterStrideCrop(size=8, stride=5) # Requires (8-1)*5 + 1 = 36 frames
    output_indices1 = transform1(indices1)
    print(f"Input length: {len(indices1)}, Output size: {len(output_indices1)}")
    print(f"Output indices (sufficient): {output_indices1}")
    # Expected: starts around index (100-36)//2 = 32. Indices: 32, 37, 42, 47, 52, 57, 62, 67

    print("-" * 20)

    # Example 2: Insufficient frames, with padding
    indices2 = list(range(0, 20)) # 20 frames
    transform2 = TemporalCenterStrideCrop(size=8, stride=5, padding=True) # Requires 36 frames
    output_indices2 = transform2(indices2)
    print(f"Input length: {len(indices2)}, Output size: {len(output_indices2)}")
    print(f"Output indices (insufficient, padded): {output_indices2}")
    # Expected: Pads [0..19] to length 36 (e.g., [0..19, 0..15]). Selects from center.

    print("-" * 20)

    # Example 3: Insufficient frames, without padding (will raise error)
    indices3 = list(range(0, 20)) # 20 frames
    transform3 = TemporalCenterStrideCrop(size=8, stride=5, padding=False) # Requires 36 frames
    try:
        output_indices3 = transform3(indices3)
        print(f"Output indices (insufficient, no padding): {output_indices3}")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("-" * 20)

    # Example 4: Small input, padding
    indices4 = [0, 1] # 2 frames
    transform4 = TemporalCenterStrideCrop(size=5, stride=3, padding=True) # Requires (5-1)*3 + 1 = 13 frames
    output_indices4 = transform4(indices4)
    print(f"Input length: {len(indices4)}, Output size: {len(output_indices4)}")
    print(f"Output indices (very insufficient, padded): {output_indices4}")
    # Expected: Pads [0, 1] to 13 (e.g., [0,1,0,1,0,1,0,1,0,1,0,1,0]). Selects from center.
