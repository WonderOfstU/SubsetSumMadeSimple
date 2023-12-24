import numpy as np

# 1D FFT 
def CirPlus_u_FFT_1dementional(S1, S2, u):
    m = max(S1) + max(S2) + 1
    fS = np.zeros(m, dtype=complex)
    fS[S1] = 1
    fT = np.zeros(m, dtype=complex)
    fT[S2] = 1
    g = np.round(np.fft.ifft(np.fft.fft(fS) * np.fft.fft(fT)).real).astype(int)
    return [x for x in range(m) if g[x] and x <= u]

# 2D FFT 
def CirPlus_u_FFT_2dementional(S1, S2, u):
    m = max(S1, key = lambda t:t[0])[0] + max(S2, key = lambda t:t[0])[0] + 1
    n = max(S1, key = lambda t:t[1])[1] + max(S2, key = lambda t:t[1])[1] + 1
    fS = np.zeros((m, n), dtype=complex)
    fT = np.zeros((m, n), dtype=complex)
    fS[tuple(zip(*S1))] = 1
    fT[tuple(zip(*S2))] = 1
    g = np.round(np.fft.ifft2(np.fft.fft2(fS) * np.fft.fft2(fT)).real).astype(int)
    return [(x, y) for x in range(m) for y in range(n) if g[x, y] and x <= u]


def AllSubsetSums_hash(S, u):
    if len(S) == 0:
        return [(0, 0)]
    elif len(S) == 1:
        return [(0, 0), (S[0], 1)]
        
    return CirPlus_u_FFT_2dementional(AllSubsetSums_hash(S[:len(S) // 2], u),\
                                      AllSubsetSums_hash(S[len(S) // 2:], u), u)


def AllSubsetSums(S, u):
    n = len(S)
    b = int(np.sqrt(n * np.log2(n)))
    
    R_l = []
    for l in range(0, b):
        S_l = [x for x in S if x % b == l]
        Q_l = [x // b for x in S_l if x // b <= u // b]
        Q_l_AllSubsetSums_hash = AllSubsetSums_hash(Q_l, u//b)
        R_l.append(list(set([z * b + j * l for z, j in Q_l_AllSubsetSums_hash if z * b + j * l <= u])))
        
    if(len(R_l) == 0):
        return [S[0] if S[0] <= u else -1]
    result = R_l[0]
    for l in range(1, b):
        result = CirPlus_u_FFT_1dementional(result, R_l[l], u)
    return result


def SubSetSumMadeSimple(S, u):
    return u in AllSubsetSums(S, u)


if __name__ == '__main__':
    # Example usage
    print(CirPlus_u_FFT_1dementional([1, 4, 5, 7, 9],[1, 4, 5, 7, 9], 100))
    #[2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18]
    print(CirPlus_u_FFT_2dementional([(1,1),(4,1),(5,1),(7,1),(9,1)], [(1,1),(4,1),(5,1),(7,1),(9,1)], 10))
    #[(2, 2), (5, 2), (6, 2), (8, 2), (9, 2), (10, 2)]
    print(AllSubsetSums_hash([1, 4, 5, 7, 9], 20))
    #[(0, 0), (1, 1), (4, 1), (5, 1), (5, 2), (6, 2), (7, 1), (8, 2), (9, 1), (9, 2), (10, 2), (10, 3), (11, 2),\
    #(12, 2), (12, 3), (13, 2), (13, 3), (14, 2), (14, 3), (15, 3), (16, 2), (16, 3), (17, 3), (17, 4), (18, 3), (19, 4), (20, 3)]
    print(AllSubsetSums([1, 4, 5, 7, 9], 100))
    #[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26]
    print(SubSetSumMadeSimple([1, 4, 5, 7, 9], 22))
    #True
    print(SubSetSumMadeSimple([1, 4, 5, 7, 9], 23))
    #False
    
    
