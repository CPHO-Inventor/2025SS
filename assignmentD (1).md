# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

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

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：



代码：

```python
def quadratic_probe_insert(keys, M):
    table = [None] * M
    result = []

    for key in keys:
        pos = key % M
        if table[pos] is None or table[pos] == key:
            table[pos] = key
            result.append(pos)
            continue

        i = 1
        instered = False
        while not instered:
            for sign in [1, -1]:
                new_pos = (pos + sign * (i ** 2)) % M
                if table[new_pos] is None or table[new_pos] == key:
                    table[new_pos] = key
                    result.append(new_pos)
                    instered = True
                    break

            i += 1

    return result


import sys

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
keys = list(map(int, data[2:2 + N]))

positions = quadratic_probe_insert(keys, M)
print(*positions)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![944cf3bb9019e0dee09b6483e472f2d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\944cf3bb9019e0dee09b6483e472f2d.png)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：



代码：

```python
from heapq import heappush,heapify,heappop

while True:
    try:
        n = int(input())
    except:
        break
    graph = {i:[] for i in range(n)}
    for i in range(n):
        list0 = [int(x) for x in input().split()]
        for j in range(n):
            if j != i:
                graph[i].append((list0[j], j))

    visit=[False]*n
    visit[0]=True
    h=graph[0]
    heapify(h)
    ans=0
    cnt=1
    while h and cnt<n:
        cost,vert=heappop(h)
        if not visit[vert]:
            visit[vert]=True
            ans+=cost
            cnt+=1
            for cn,nbr in graph[vert]:
                if not visit[nbr]:
                    heappush(h,(cn,nbr))
    print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![02fe0d9133e034c1385c8f6350b1580](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\02fe0d9133e034c1385c8f6350b1580.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：



代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist=[float('inf') for i in range(n)] 
        dist[src] = 0
        res = float('inf')
        for i in range(k + 1):
            dist1 = [float('inf') for i in range(n)]
            for [a, b, w] in flights:
                dist1[b] = min(dist1[b], dist[a] + w)
            dist = dist1
            res = min(res, dist[dst])
        return res if res != float('inf') else -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![591fed64566cf5529f4ff71980a0439](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\591fed64566cf5529f4ff71980a0439.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：



代码：

```python
import heapq
n, m = map(int, input().split())
graph = [[] for i in range(n + 1)]
for i in range(m):
    a, b, w = map(int, input().split())
    graph[a].append([w, b])

h = [(0, 1)]
d = [float('inf')] * (n + 1)
d[1] = 0
heapq.heapify(h)
while h:
    dist, vert = heapq.heappop(h)
    if d[vert] < dist:
        continue
    for d0, vert0 in graph[vert]:
        if d[vert0] > d[vert] + d0:
            d[vert0] = d[vert] + d0
            heapq.heappush(h, (d[vert0], vert0))
print(d[-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![8d4753757c26fd4b452b8db3ca9661a](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\8d4753757c26fd4b452b8db3ca9661a.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

这两天和同学把图给复习完了，还在写作业，先把写完的部分发上去（怎么这个学期总在ddl之前交作业，nao）











