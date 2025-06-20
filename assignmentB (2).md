# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

2025 spring, Complied by <mark>董天泽 物理学院</mark>



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：



代码：

```python
from collections import deque

def inside(x, y, a, b):
    return 0 <= x < a and 0 <= y < b

def bfs(start, end, m, a, b):
    queue = deque()
    start_x, start_y = start
    queue.append([start_x, start_y, 0])

    m[start_x] = m[start_x][:start_y] + 'V' + m[start_x][start_y+1:]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        x, y, steps = queue.popleft()
        if [x, y] == end:
            return steps
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if inside(nx, ny, a, b) and m[nx][ny] != 'V' and m[nx][ny] != '#':
                # Mark as visited before adding to queue
                m[nx] = m[nx][:ny] + 'V' + m[nx][ny+1:]
                queue.append([nx, ny, steps + 1])
    return -1

t = int(input())
for _ in range(t):
    a, b = map(int, input().split())
    m = []
    start = None
    end = None
    for i in range(a):
        s = input().strip()
        m.append(s)
        if 'S' in s:
            start = [i, s.find('S')]
        if 'E' in s:
            end = [i, s.find('E')]

    grid = [list(row) for row in m]
    grid = [''.join(row) for row in grid]

    ans = bfs(start, end, grid, a, b)
    print(ans if ans != -1 else "oop!")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![d280ad1a52842c6178a1b435b165c16](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\d280ad1a52842c6178a1b435b165c16.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：



代码：

```python
lass Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        id = [0] * n  
        for i in range(1, n):
            id[i] = id[i - 1]
            if nums[i] - nums[i - 1] > maxDiff:
                id[i] += 1  

        return [id[u] == id[v] for u, v in queries]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![9c90f1cd6ab4e9178b03b0af819f674](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\9c90f1cd6ab4e9178b03b0af819f674.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：



代码：

```python
grade = [float(x) for x in input().split()]
le = len(grade)
grade.sort()
targ = grade[int(le * 0.4)]
left = 0
right = 1000000000
while left <= right:
    mid = (left + right) // 2
    gd = targ * mid / 1000000000 + 1.1 ** (targ * mid / 1000000000)
    if gd >= 85:
        right = mid - 1
    else:
        left = mid + 1
print(left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![8ca1cdab62c1536bad99e28a528c166](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\8ca1cdab62c1536bad99e28a528c166.png)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：



代码：

```python
def is_cyclic(u, graph, vis):
    vis[u] = 0  
    for v in graph[u]:
        if vis[v] == -1:  
            if is_cyclic(v, graph, vis):
                return True
        elif vis[v] == 0:  
            return True
    vis[u] = 1  
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    graph = [[] for _ in range(n)]
    vis = [-1] * n  
    
    index = 2
    for _ in range(m):
        u = int(data[index])
        v = int(data[index + 1])
        graph[u].append(v)
        index += 2
    
    for i in range(n):
        if vis[i] == -1:  
            if is_cyclic(i, graph, vis):
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![8ed50022d4e782f85759d9287b990e1](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\8ed50022d4e782f85759d9287b990e1.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：



代码：

```python
import heapq
from collections import defaultdict

def dijkstra_point_to_point(graph, start, end):

    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    path = {point: None for point in graph.keys()}
    path[start] = [[start, 0]]

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                path[neighbor] = path[current_node] + [[neighbor, weight]]

    return distances[end], path[end]

def trans(path):
    ans = ""
    for p in path[1:]:
        ans += f"->({p[1]})->" + p[0]
    return path[0][0] + ans


p = int(input())
points = [input().strip() for _ in range(p)]
graph = defaultdict(dict)
for _ in range(int(input())):
    a, b, d = input().split()
    d = int(d)
    graph[a][b] = d
    graph[b][a] = d

for _ in range(int(input())):
    start, end = input().split()
    _, path = dijkstra_point_to_point(graph, start, end)
    print(trans(path))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![2cec7167c3ef2fc75b0bf4e37fe0989](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\2cec7167c3ef2fc75b0bf4e37fe0989.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：



代码：

```python
import sys

def is_valid_move(x, y, board, n):
    return 0 <= x < n and 0 <= y < n and board[x][y] == -1

def get_degree(x, y, board, n, moves):
    count = 0
    for dx, dy in moves:
        if is_valid_move(x + dx, y + dy, board, n):
            count += 1
    return count

def knights_tour_warnsdorff(n, sr, sc):
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]
    board = [[-1 for _ in range(n)] for _ in range(n)]
    board[sr][sc] = 0
    
    def backtrack(x, y, move_count):
        if move_count == n * n:
            return True
        
        next_moves = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny, board, n):
                degree = get_degree(nx, ny, board, n, moves)
                next_moves.append((degree, nx, ny))
        
        next_moves.sort()  # 按 Warnsdorff 规则选择最少可行移动的方向
        
        for _, nx, ny in next_moves:
            board[nx][ny] = move_count
            if backtrack(nx, ny, move_count + 1):
                return True
            board[nx][ny] = -1  # 回溯
        
        return False
    
    if backtrack(sr, sc, 1):
        print("success")
    else:
        print("fail")

if __name__ == "__main__":
    n = int(sys.stdin.readline().strip())
    sr, sc = map(int, sys.stdin.readline().strip().split())
    knights_tour_warnsdorff(n, sr, sc)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![c7c6acaf7b0ede0d264b5d7f8a37865](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\c7c6acaf7b0ede0d264b5d7f8a37865.png)



## 2. 学习总结和收获

本来以为作业是下周二交，赶紧补完，还在补树那块的知识。











