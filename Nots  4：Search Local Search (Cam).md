# Nots  4：Search: Local Search (Cam)

## Local Search（局部搜索）

In the previous note, we wanted to find the goal state, along with the optimal path to get there. But in some problems, we only care about finding the goal state — reconstructing the path can be trivial. For example, in Sudoku, the optimal configuration is the goal. Once you know it, you know to get there by filling in the squares one by one.  

Local search algorithms allow us to find goal states without worrying about the path to get there. In local search problems, the state space are sets of "complete" solutions. We use these algorithms to try to find a configuration that satisfies some constraints or optimizes some objective function.

以上的大致意思就是不同于上节的贪心搜索与A*算法，本节中的算法所解决的问题不需要获得从初始状态到目标状态的路径，仅需要目标状态（一个最优解），局部搜索可以解决这个问题。

局部搜索的的基本思想：

- 初始解：从状态空间中随机选择或指定一个初始状态。
- 邻域搜索：在当前解的邻域内搜索所有可能的解（邻居解）。邻居是指与当前解只有一个小变化的解。
- 选择更优解：从邻居解中选择一个比当前解更优的解作为新的当前解。如果没有更优的邻居解，则可能终止搜索，或者根据算法的变种采取不同策略。
- 迭代：重复邻域搜索和选择更优解的步骤，直到满足某个终止条件（如达到最大迭代次数或找到满意的解）。

## Hill-Climbing Search（爬山搜索）

在笔者的理解里，爬上搜索可以人为就是最基础的局部搜索算法，承袭了局部搜索的基本思想。

爬山搜索的局限性：容易陷入局部最优解，无法找到全局最优解；如果当前解的邻居都不能更新解，算法将陷入停滞。

##  Simulated Annealing Search（模拟退火算法）

模拟退火算法是爬山搜索的变种，他优化了爬山搜索。

模拟退火算法的基本思想

- 初始解和初始温度：从解空间中随机选择一个初始解，并设置一个较高的初始温度。
- 邻域搜索：在当前解的邻域内随机选择一个新解
- 接受准则（是否将邻域搜索得来的解作为下一次迭代的初始解）：
  - 如果当前新解优于初始解，则接受新解
  - 如果新解比当前解差，则以一定的概率接受新解，这个概率由当前温度决定。温度越高，接受较差解的概率越大，随着温度降低，接受较差解的概率逐渐减小。
- 降温策略：逐步降低温度，使得算法在初期可以跳出局部最优解，而在后期能够稳定收敛到全局最优解。

算法的优点以及缺点：

优点：

- 全局优化能力强：可以跳出局部最优，找到全局最优。
- 简单易实现：算法思想简单，实现简单。

缺点：

- 参数敏感：初始温度，降温速率等参数需要仔细调整。
- 计算量大：需要大量迭代，计算时间较长。

伪代码：

```python
import numpy as np

def simulated_annealing(func, x0, T0, alpha, stopping_T, stopping_iter):
    x = x0
    T = T0
    current_energy = func(x)
    best_x = x
    best_energy = current_energy
    iteration = 0

    while T > stopping_T and iteration < stopping_iter:
        x_new = x + np.random.uniform(-1, 1, size=x.shape)  # 随机生成邻居解
        new_energy = func(x_new)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / T):
            x = x_new
            current_energy = new_energy

            if new_energy < best_energy:
                best_x = x_new
                best_energy = new_energy

        T = T * alpha  # 降温
        iteration += 1

    return best_x, best_energy

# 示例目标函数
def objective_function(x):
    return x**2 + 4*np.sin(5*x) + 0.1*x**4

# 参数设置
initial_solution = np.array([2.0])
initial_temperature = 1000
cooling_rate = 0.99
stopping_temperature = 1e-3
stopping_iteration = 10000

best_solution, best_energy = simulated_annealing(objective_function, initial_solution, initial_temperature, cooling_rate, stopping_temperature, stopping_iteration)
print("Best solution:", best_solution)
print("Best energy:", best_energy)

```

## Local Beam Search（局部光束搜索）

Local Beam Search的核心思想是，从多个初始解开始，在每次迭代中生成这些解的所有邻居，并从中选择最好的几个解作为新的候选解。这个过程不断重复，直到满足停止条件（如达到最大迭代次数或找到满意的解）。

## Genetic Algorithms（遗传算法）

算法大致有以下步骤：

1. 为种群中的每一个个体计算Value（通过某一函数）
2. 根据每个个体的Value确定遗传概率
3. 交配生成下一代（有多种交配方式，按值交换、局部交换等）
4. 变异

这个算法笔者在准备研究生复试时候学过，暂时不计太多。