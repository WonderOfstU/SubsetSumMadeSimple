import numpy as np    
from scipy.optimize import fsolve
import os
import glob

def time_complexity(n, u):
    return np.sqrt(n * np.log2(n)) * u * np.log2(u)

def generate_contrast(size): #生成SubsetSumMadeSimple与dp的运行时间对比数据文件，size为生成文件数量
    sequence_n = np.linspace(size * 20, size * 100, num=size, dtype=int)
    print(sequence_n)

    fileNames = []
    for i in range(size):
        n = sequence_n[i]
        m = n * np.sqrt(np.log2(n)).astype(int)
        S = np.random.choice(range(1, m+1), size=n, replace=False).astype(np.int64)
        u = int(np.random.uniform(max(S) + min(S) // 2, sum(S) + 1))
        filename = f'./data/SubsetSum_contrast-self_generated-{i + 1:02}-{n}-{u}.txt'
        with open(filename, 'w') as file:
            file.write(f"{n} {u}\n")
            file.write(' '.join(map(str, S)))
        fileNames.append(filename)
    return fileNames


def generate_TimeComplexity_test(size): #生成时间复杂度测试数据文件，size为生成文件数量
    sequence_n = np.linspace(100, 4096, num=size, dtype=int)
    sequence_tc = np.linspace(10000000, 80000000, num=size, dtype=int)

    fileNames = []
    for i in range(size):
        n = sequence_n[i]
        tc = sequence_tc[i]
        m = n * np.sqrt(np.log2(n)).astype(int) 
        S = np.random.choice(range(1, m+1), size=n, replace=False).astype(np.int64)
        g = lambda u: time_complexity(n, u) - tc
        u = int(fsolve(g, tc // np.sqrt(n * np.log2(n))))
        filename = f'./data/SubsetSum_TimeComplexity_test-self_generated-{i + 1:003}-{n}-{u}.txt'
        with open(filename, 'w') as file:
            file.write(f"{n} {u}\n")
            file.write(' '.join(map(str, S)))
        fileNames.append(filename)
    return fileNames


if __name__ == '__main__':
    # Example usage
    conservation_key = input('Erase all data files and regenerate them?\nType Yes if you want to do so.\n')
    if conservation_key == 'Yes': #若想执行，请在命令行中输入Yes
        files = glob.glob('./data/*')
        for f in files:
            os.remove(f)
        fileNames1 = generate_TimeComplexity_test(size=9)
        fileNames2 = generate_contrast(size=256)
        print(fileNames1,'\n',fileNames2)

