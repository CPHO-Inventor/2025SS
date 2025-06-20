# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Complied by <mark>董天泽 物理学院</mark>



> **说明：**
>
> 1. **⽉考**：AC?<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
> 2. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：



代码：

```python
n, k = map(int, input().split())
list0 = []
for i in range(n):
    a, b = map(int, input().split())
    list0.append([a, b, i + 1])
list0.sort()
list1 = list0[(n - k):]
for i in range(len(list1)):
    list1[i][0], list1[i][1] = list1[i][1], list1[i][0]

list1.sort()
print(list1[-1][-1])

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![c06822490bd9166cb2c973399e85de4](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\c06822490bd9166cb2c973399e85de4.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：



代码：

```python
n = int(input())
from math import comb
print(comb(2 * n, n) // (n + 1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![4a2361e96288060ceea4c7ffcd55fb4](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\4a2361e96288060ceea4c7ffcd55fb4.png)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：



代码：

```python
n = int(input())
list0 = [str(x) for x in input().split()]
queue_num = [[],[],[],[],[],[],[],[],[]]
queue_let = [[],[],[],[]]
for i in list0:
    num0 = int(i[1])
    queue_num[num0 - 1].append(i)

for i in range(9):
    for j in queue_num[i]:
        let0 = j[0]
        if let0 == 'A':
            queue_let[0].append(j)
        elif let0 == 'B':
            queue_let[1].append(j)
        elif let0 == 'C':
            queue_let[2].append(j)
        elif let0 == 'D':
            queue_let[3].append(j)

for i in range(9):
    print(f'Queue{i + 1}:' + ' '.join(queue_num[i]))
list1 = []
for i in range(4):
    if i == 0:
        print('QueueA:' + ' '.join(queue_let[0]))
        list1 += queue_let[0]
    elif i == 1:
        print('QueueB:' + ' '.join(queue_let[1]))
        list1 += queue_let[1]
    elif i == 2:
        print('QueueC:' + ' '.join(queue_let[2]))
        list1 += queue_let[2]
    elif i == 3:
        print('QueueD:' + ' '.join(queue_let[3]))
        list1 += queue_let[3]

print(' '.join(list1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![a43fab69a1b0ac5ed57cebcc14328c9](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a43fab69a1b0ac5ed57cebcc14328c9.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：



代码：

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：



代码：

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def build_complete_tree(level_order):
    nodes = [TreeNode(val) for val in level_order]

    n = len(nodes)
    for i in range(n):
        left_idx = 2 * i + 1
        if left_idx < n:
            nodes[i].left = nodes[left_idx]

        right_idx = 2 * i + 2
        if right_idx < n:
            nodes[i].right = nodes[right_idx]

    return nodes[0]

def solve(root):
    if not root:
        return 0
    
    b = root.val
    a = solve(root.left) + solve(root.right)
    
    if root.left:
        b += solve(root.left.right) + solve(root.left.left)
    if root.right:
        b += solve(root.right.left) + solve(root.right.right)
    
    return max(a, b)

n = int(input())
Nodes = [int(_) for _ in input().split()]
root = build_complete_tree(Nodes)

print(solve(root))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![25ab380dba09e5e58566d71d5281198](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\25ab380dba09e5e58566d71d5281198.png)



## 2. 学习总结和收获

最近在赶大作业，所以这次作业也是在ddl前赶完，就只做了自己做出来的部分，剩下的部分补了课之后再做。感觉只能AC3-4了，有点慌。不知道这次月考和最后上机考试相比难度如何。











