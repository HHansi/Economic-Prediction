# Created by Hansi at 12/5/2020

# Python 3 implementation to find
# the number closest to n

# Function to find the number closest
# to n and divisible by m
def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if ((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)):
        return n1

    # else n2 is the required closest number
    return n2


closestNumber(13, 4)


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


def create_future_quarter_labels(quarter, n):
    """
    Create future quarter labels formatted as 2020 Q3
    :param quarter: str
        Last quarter label
    :param n: int
        Number of future labels required
    :return: list of str
        Future labels
    """
    labels = []
    year = int(quarter.split()[0])
    q_num = int(quarter.split()[1][1])

    for i in range(0, n):
        new_q = q_num + 1
        if new_q > 4:
            year = year + 1
            new_q = 1
        q_num = new_q
        label = f'{year} Q{q_num}'
        labels.append(label)
    return labels

