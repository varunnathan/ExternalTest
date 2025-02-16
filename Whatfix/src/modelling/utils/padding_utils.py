def pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d[:width] + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def pad_3d(data, pad_id, dim=1, width=-1):
    #dim = 1 or 2
    if dim < 1 or dim > 2:
        return data
    if (width == -1):
        if (dim == 1):
            #dim 0,2 is same across the batch
            width = max(len(d) for d in data)
        elif (dim == 2):
            #dim 0,1 is same across the batch
            for entry in data:
                width = max(width, max(len(d) for d in entry))
        #print(width)
    if dim == 1:
        rtn_data = [d[:width] + [[pad_id] * len(data[0][0])] * (width - len(d)) for d in data]
    elif dim == 2:
        rtn_data = []
        for entry in data:
            rtn_data.append([d[:width] + [pad_id] * (width - len(d)) for d in entry])
    return rtn_data