# def get_atsc(raw_array: np.ndarray, config: dict) -> Tuple[Optional[List[int]], np.ndarray]:
#     # filter by person d^2 = 2m(1-r)
#     m = config["m"]
#     threshold_d = config["threshold_d"]
    
#     # Matrix Profile 
#     mp = stumpy.stump(raw_array, m=m)

#     distances = np.copy(mp[:, 0])
#     zero_mask = distances <= 1e-5
#     if np.all(zero_mask):
#         return None, np.array([])

#     distances[zero_mask] = np.inf
#     anchor_idx = int(np.argmin(distances))
#     v_d = distances[anchor_idx]
    
#     if v_d > threshold_d or np.isinf(v_d):
#         return None, np.array([])

#     left_I = np.copy(mp[:, 2])   
#     invalid_mask = (mp[:, 0] > threshold_d) | zero_mask
#     left_I[invalid_mask] = -1  

#     # Cycle Detection
#     backward_chain = []
#     curr_left = left_I[anchor_idx]
#     visited = set()

#     while curr_left != -1 and curr_left not in visited:
#         visited.add(curr_left)
#         backward_chain.append(curr_left)
#         curr_left = left_I[curr_left]
        
#     backward_chain.reverse() 
#     atsc_chain = backward_chain + [anchor_idx]
    
#     atsc_chain_v = np.array([raw_array[idx : idx + m] for idx in atsc_chain])
#     return atsc_chain, atsc_chain_v
