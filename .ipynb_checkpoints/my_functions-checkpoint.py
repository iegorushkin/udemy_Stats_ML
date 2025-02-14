import numpy as np

def rank_transform(data):
    data_copy = data.copy()
    data_len = len(data)
    data_rank = np.zeros(len(data), dtype=int)  # output, rank-transformed vector

    # loop through every index value of data     
    for i in range(data_len):
        cur_position, cur_step = 0, 1 
        # The main idea is that we compare the value of the data
        # in the cur_position + cur_step cell with the value in cur_position.
        # The sum of these indices can't be > the length of the dataset
        while cur_position + cur_step < data_len:
            # If the value in the cell with index cur_position + cur_step is
            # less than the value in the cell with index cur_position,
            # we shift cur_position and reset the step.
            if (data_copy[cur_position + cur_step] < data_copy[cur_position]
                or np.isnan(data_copy[cur_position])): 
                # ^ change position if data_copy[cur_position] is NaN
                cur_position = cur_position + cur_step
                cur_step = 1
            # If the value in the cell with index cur_position + cur_step is
            # greater than (or equal to) the value in the cell with index cur_position,
            # we increase the step and perform a new comparison.
            elif (data_copy[cur_position + cur_step] >= data_copy[cur_position]
                  or np.isnan(data_copy[cur_position + cur_step])):
                # ^ increase step if data_copy[cur_position + cur_step] is NaN
                cur_step += 1
        # Having found the index of the minimum element in the data,
        # we declare the value corresponding to it as NaN.
        # This allows us to impose additional conditions in the if-else loop,
        # which ensure finding successively increasing values in the data and their indices. 
        data_copy[cur_position] = np.nan
        # The result of current iteration of the rank-transform.
        data_rank[cur_position] = int(i)

    return data_rank