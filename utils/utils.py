def get_column(img, col):
    return img[:, col]


def get_row(img, row):
    return img[row, :]


def get_segment(img, x1, y1, x2, y2):
    if x1==x2 and y1==y2:
        return img[x1, y1]
    if x1==x2:
        return img[y1:y2, x1]
    if y1==y2:
        return img[y1, x1:x2]
    Y = [int(x*(y2-y1)//(x2-x1)+y1) for x in range(x2-x1)]
    X = [i for i in range(x1,x2)] 
    return img[Y,X]

