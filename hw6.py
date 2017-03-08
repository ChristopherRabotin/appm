from numpy import log

if __name__ == '__main__':
    # load the data file
    f = open('data/hw6/spambase.data')
    data = []
    for line in f.readlines():
        ldt = [float(v) for v in line.split(',')]
        # Change the label from {1, 0} to {1, -1}.
        if ldt[-1] == 0:
            ldt[-1] = -1
        # Normalize the data.
        ldt[:-1] = [log(xij + 0.1) for xig in ldt[:-1]]
        data.append(ldt)
