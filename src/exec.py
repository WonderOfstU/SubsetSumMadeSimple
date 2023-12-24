import numpy as np
import generation
import SubSetSumOriginal_dp
import SubSetSumMadeSimple_FFT_DivideAndConquer
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
import os
import glob


def time_complexity(n, u):
    return np.sqrt(n * np.log2(n)) * u * np.log2(u)

def time_complexity_ab(nu, a, b):
    n, u = nu
    return a * time_complexity(n, u) + b

def analyze_contrast(Datas): #进行SubsetSumMadeSimple与dp的运行时间对比
    txt = []
    i = 1
    l = len(Datas)
    for dat in Datas:
        print(f'From SubsetSum_contrast-self_generated-{i:02}-{dat[0]}-{dat[1]}.txt:')
        txt.append(f'From SubsetSum_contrast-self_generated-{i:02}-{dat[0]}-{dat[1]}.txt:')
        
        start_time = time.time()
        print(f'MadeSimple_ans: {SubSetSumMadeSimple_FFT_DivideAndConquer.SubSetSumMadeSimple(dat[2], dat[1])}')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'MadeSimple_exec_time: {execution_time}s')
        txt.append(f'MadeSimple_exec_time: {execution_time}s')
        
        start_time = time.time()
        print(f'dp_ans: {SubSetSumOriginal_dp.SubSetSumOriginal(dat[2], dat[1])}')
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'dp_exec_time: {execution_time}s')
        txt.append(f'dp_exec_time: {execution_time}s')

        print(f'{i}/{l} completed')
        i += 1
        
    np.savetxt('./experiments/Run_time_contrast_between_MadeSimple_and_dp.txt', txt, fmt='%s')


def cal_y(S, u, y, i, text=''): 
    start_time = time.time()
    print(SubSetSumMadeSimple_FFT_DivideAndConquer.SubSetSumMadeSimple(S, u))
    end_time = time.time()
    execution_time = end_time - start_time
    y[i] += execution_time
    print(text + f'{execution_time}')


def draw_UnivariateNonlinearRegression(function, x, y, title, xlabel, ylabel, saveLocation): #一元非线性回归图像绘制
    plt.clf()
    initial_guess = [1 for _ in range(len(function.__code__.co_varnames) - 1)]
    params, _ = curve_fit(function, x, y, p0=initial_guess)
    plt.xlim([min(x), max(x)])
    plt.ylim([min(y), max(y)])
    plt.rcParams["figure.dpi"] = 2560
    plt.scatter(x, y, s=44, marker='.', color='blue', alpha=0.8, label='Test Data')
    plt.plot(x, function(x, *params), color='r', label='Fitted Curve')
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", color="gray", linewidth=0.75)
    plt.grid(True, which="minor", linestyle=":", color="lightgray", linewidth=0.75)
    plt.xlabel(xlabel,fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title,fontsize=16)
    plt.legend()
    plt.savefig(saveLocation, dpi=2560)


def analyze_TimeComplexity_total(Datas, round = 2): #进行总体时间复杂度分析，绘制二维图像
    x = np.asarray([time_complexity(n, u) for n, u, _ in Datas])
    y = np.zeros(x.size)
    
    for _ in range(round):
        i = 0
        l = len(Datas)
        for n, u, S in Datas:
            cal_y(S, u, y, i, f'SubsetSum_TimeComplexity_test-self_generated-{i + 1:003}:\
                \ntime_complexity(n, u), execution_time: {time_complexity(n, u)}, ')
            print(f'{i+1}/{l} completed  round{_+1}/{round}')
            i += 1
    y /= round
    np.savetxt(f'./experiments/Time_complexity_analysis_of_MadeSimple_total-{len(Datas)}.txt', y)
    func_linear = lambda x, a, b: a * x + b
    draw_UnivariateNonlinearRegression(function=func_linear, x=x, y=y,\
            title=r'$ExecutionTime-n&u \quad t(n,u)=a\sqrt{n \log n} u \log u + b$',\
            xlabel='$\sqrt{n \log n} u \log u \; (1e7)$',\
            ylabel='t (second)',\
            saveLocation=f'./experiments/Time_complexity_analysis_of_MadeSimple_total-{len(Datas)}.png')
    
    
def analyze_TimeComplexity_separate(Data, size_n=256, size_u=256, round=2, step_3d=20): #分别对n和u进行时间复杂度分析
    dn, du, dS = Data
    dS = np.asarray(dS)
    #对n进行时间复杂度分析，绘制二维图像
    sequence_n = np.linspace(dn//10, dn, num=size_n, dtype=int)
    x = sequence_n
    y = np.zeros(x.size)
    for _ in range(round):
        i = 0
        for n in sequence_n:
            m = n * np.sqrt(np.log2(n)).astype(int) 
            S = dS[dS<=m]
            S = S[:n]
            cal_y(S, du, y, i, f'constu={du}, n={n}: ')
            print(f'{i+1}/{size_n} completed  round{_+1}/{round}')
            i += 1
    y /= round
    np.savetxt(f'./experiments/Time_complexity_analysis_of_MadeSimple_separate_constu={du}_n-{size_n}.txt', y)
    func_n = lambda n, a, b: a * np.sqrt(n * np.log2(n)) + b
    draw_UnivariateNonlinearRegression(function = func_n, x=x, y=y,\
            title=r'$ExecutionTime-n \quad t(n)= a \sqrt{n \log n} + b$',\
            xlabel='n',\
            ylabel='t (second)',\
            saveLocation=f'./experiments/Time_complexity_analysis_of_MadeSimple_separate_constu={du}_n-{size_n}.png')
    #对u进行时间复杂度分析，绘制二维图像
    sequence_u = np.linspace(du//10, du, num=size_u, dtype=int)
    x = sequence_u
    y = np.zeros(x.size)
    for _ in range(round):
        i = 0
        m = dn * np.sqrt(np.log2(dn)).astype(int)
        S = dS
        for u in sequence_u:
            cal_y(S, u, y, i, f'constn={dn}, u={u}: ')
            print(f'{i+1}/{size_u} completed  round{_+1}/{round}')
            i += 1
    y /= round
    np.savetxt(f'./experiments/Time_complexity_analysis_of_MadeSimple_separate_constn={dn}_u-{size_u}.txt', y)
    func_u = lambda u, a, b: a * u * np.log2(u) + b
    draw_UnivariateNonlinearRegression(function=func_u, x=x, y=y,\
            title=r'$ExecutionTime-u \quad t(u)= a u \log u + b$',\
            xlabel='u',\
            ylabel='t (second)',\
            saveLocation=f'./experiments/Time_complexity_analysis_of_MadeSimple_separate_constn={dn}_u-{size_u}.png')
    #对n和u进行时间复杂度分析，绘制三维图像
    sequence_n = np.linspace(dn//8, dn, num=step_3d, dtype=int)
    sequence_u = np.linspace(du//8, du, num=step_3d, dtype=int)
    t = np.zeros((step_3d, step_3d))
    for _ in range(round):
        for i in range(sequence_n.size):
            n = sequence_n[i]
            m = n * np.sqrt(np.log2(n)).astype(int) 
            S = dS[dS<=m]
            S = S[:n]
            for j in range(sequence_u.size):
                u = sequence_u[j]
                start_time = time.time()
                SubSetSumMadeSimple_FFT_DivideAndConquer.SubSetSumMadeSimple(S, u)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f'n={n}, u={u}, execution_time={execution_time}')
                print(f'{i * step_3d + j + 1}/{step_3d**2} completed  round{_+1}/{round}')
                t[(i, j)] += execution_time
    t /= round
    np.savetxt(f'./experiments/Time_complexity_analysis_of_MadeSimple_separate_n_u-{step_3d}.txt', t, fmt='%21.18e')
    n, u = np.meshgrid(sequence_n, sequence_u, indexing='ij')
    
    n_flat = n.flatten()
    u_flat = u.flatten()
    t_flat = t.flatten()

    initial_guess = [1, 1]
    params, _ = curve_fit(time_complexity_ab, (n_flat, u_flat), t_flat, p0=initial_guess)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    angles = [(0, -90), (0, 0), (-1, -30), (18, -111), (6, -156), (13, 114)]

    ax.scatter(n_flat, u_flat, t_flat, c='y', marker='o', label='Test Data', alpha=1, s=4)
    ax.plot_surface(n, u, time_complexity_ab((n, u), *params).reshape(n.shape), color='r', alpha=0.8, edgecolor='none', label='Fitted Surface', cmap='coolwarm', rstride=1, cstride=1)
    ax.set_xlabel('n')
    ax.set_ylabel('u')
    ax.set_zlabel('ExecutionTime (second)')
    ax.set_title(f'ExecutionTime-n&u{i+1}')
    ax.legend()
    for i, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f'./experiments/Time_complexity_analysis_of_MadeSimple_separate_n_u_angle{i + 1}-{step_3d}.png')

    
def read_data(fileNames):   #读取数据文件
    Datas = []
    for file_name in fileNames:
        with open(file_name, 'r') as file:
            first_line = file.readline().split()
            n = int(first_line[0])
            u = int(first_line[1])
            S = [int(i) for i in file.read().split()]
            Datas.append([n, u, S])
    return Datas
    

if __name__ == '__main__':
    conservation_key = input('Erase all data&experment files and regenerate them then execute the experiment?\nType Yes if you want to do so.\n')
    if conservation_key == 'Yes': #若想执行，请在命令行中输入Yes
        files = glob.glob('./data/*') + glob.glob('./experiments/*') #删除所有数据文件和实验结果文件
        for f in files:
            os.remove(f)
    
        start_time = time.time()
        
        #生成SubsetSumMadeSimple与dp的运行时间对比数据文件，size为目标数据文件个数
        fileNames = generation.generate_contrast(size=10) 
        #执行测试
        analyze_contrast(read_data(fileNames)) 
        #生成SubsetSumMadeSimple时间复杂度分析数据文件，size为目标数据文件个数
        fileNames = generation.generate_TimeComplexity_test(size=512) 
        #执行测试，round为测试轮数，每个测试点对应一个测试文件，运行round次取平均值，测试点个数为前一行的size，即对应数据文件的个数
        analyze_TimeComplexity_total(read_data(fileNames), round=4) 
        #执行测试，size_n、size_u、step_3d**2为相应测试点个数，round为测试轮数，每个测试点运行round次取平均值
        analyze_TimeComplexity_separate(read_data(fileNames[-1:])[0], size_n=512, size_u=512, step_3d=20, round=4) 
        
        end_time = time.time()
        
        print(f'Total execution time: {end_time - start_time}s') #输出整个测试运行时间
        




    
        
