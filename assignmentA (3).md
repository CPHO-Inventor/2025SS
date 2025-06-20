# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

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

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

要求创建Graph, Vertex两个类，建图实现。

思路：



代码：

```python
n, m = map(int, input().split())
ans = [[0 for i in range(n)] for j in range(n)]
for i in range(m):       
    knot1, knot2 = map(int, input().split())       
    ans[knot1][knot1] += 1       
    ans[knot2][knot2] += 1       
    ans[knot1][knot2] -= 1       
    ans[knot2][knot1] -= 1
for j in range(n):       
    print(' '.join(map(str, ans[j])))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![5f8174c80dcecf2f3a0403ecf11ed75](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\5f8174c80dcecf2f3a0403ecf11ed75.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：



代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
        return res


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![3fbd583ce8528f20f9546c2b438c153](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\3fbd583ce8528f20f9546c2b438c153.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：



代码：

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: return []
        phone = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        queue = ['']  
        for digit in digits:
            for _ in range(len(queue)):
                tmp = queue.pop(0)
                for letter in phone[ord(digit)-50]:
                    queue.append(tmp + letter)
        return queue
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![7690f96dbb5b9c1de8835bd3938a01f](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\7690f96dbb5b9c1de8835bd3938a01f.png)



### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：



代码：

```python
class sepTreeNode:
    def __init__(self):
        self.is_leaf = False
        self.children = [None for _ in range(10)]


t = int(input())
for _ in range(t):
    n = int(input())
    numbers = [input() for _ in range(n)]
    numbers.sort(key=lambda x: len(x))
    flag = False

    root = sepTreeNode()
    for number in numbers:
        if flag:
            break

        cur = root
        for i, d in enumerate(number):
            d = int(d)
            if not cur.children[d]:
                cur.children[d] = sepTreeNode()
            if cur.children[d].is_leaf:
                flag = True
                break
            if i == len(number) - 1:
                cur.children[d].is_leaf = True

            cur = cur.children[d]

    print("YES" if not flag else "NO")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![129b9897d42f665ca286732dff6b163](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\129b9897d42f665ca286732dff6b163.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：



代码：

```python
from collections import deque

def get_neighbors(word, words_set):
    neighbors = []
    chars = list(word)
    for i in range(4):
        original_char = chars[i]
        if original_char.islower():
            alphabet = 'abcdefghijklmnopqrstuvwxyz'
        else:
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for c in alphabet:
            if c == original_char:
                continue
            temp = chars.copy()
            temp[i] = c
            new_word = ''.join(temp)
            if new_word in words_set:
                neighbors.append(new_word)
    return neighbors

n = int(input())
words = [input().strip() for _ in range(n)]
words_set = set(words)
start, end = input().strip().split()

if start not in words_set or end not in words_set:
    print("NO")
    exit()

if start == end:
    print(start)
    exit()

queue = deque([start])
predecessor = {start: None}
found = False

while queue:
    current = queue.popleft()
    if current == end:
        found = True
        break
    neighbors = get_neighbors(current, words_set)
    for neighbor in neighbors:
        if neighbor not in predecessor:
            predecessor[neighbor] = current
            queue.append(neighbor)

if not found:
    print("NO")
else:
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()
    print(' '.join(path))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![a2c165ca14fa5a03d6a619ea4af583b](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a2c165ca14fa5a03d6a619ea4af583b.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：



代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        path = [0] * n
        col = [False] * n 
        m = 2 * n
        diag1 = [False] * m 
        diag2 = [False] * m 

   
        def dfs(r):
            if r == n:
                res.append(['.' * i + 'Q' + '.' * (n-i-1) for i in path])
                return
            for c in range(n):
                if not col[c] and not diag1[r-c] and not diag2[r+c]:
                    path[r] = c
                    col[c] = diag1[r-c] = diag2[r+c] = True
                    dfs(r + 1)
                    col[c] = diag1[r-c] = diag2[r+c] = False

        dfs(0)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![a2bfde6c39edb50166386f240c581a9](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a2bfde6c39edb50166386f240c581a9.png)



## 2. 学习总结和收获

还在补课，补每日选做，对数算机考充满了恐慌（











