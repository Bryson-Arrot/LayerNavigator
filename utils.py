import numpy
import string

L2N = {"A":0,"B":1,"C":2,"D":3,"E":4}
YN2N = {"Yes":0,"No":1}

def random_str():
    return "".join(numpy.random.choice(list(string.ascii_lowercase),5)).capitalize()

def terminal_here():
    import code; code.interact(local=locals())

def short_float(x,prec=3):
    return f"{x:.{prec}f}"

def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs_seq = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            lcs_seq.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs_seq.reverse()

    return dp[m][n], lcs_seq