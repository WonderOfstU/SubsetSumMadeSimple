def SubSetSumOriginal(S, u):
    n = len(S)
    dp = [[False for i in range(u+1)] for j in range(n+1)]
    for i in range(n+1):
        dp[i][0] = True
    for i in range(1, n+1):
        for j in range(1, u+1):
            if S[i-1] <= j:
                dp[i][j] = dp[i-1][j-S[i-1]] or dp[i-1][j]
            else:
                dp[i][j] = dp[i-1][j]
    return dp[n][u]

if __name__ == '__main__':
    # Example usage
    print(SubSetSumOriginal([1, 4, 5, 7, 9], 10))
    # True

    