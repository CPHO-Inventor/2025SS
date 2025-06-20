# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Complied by <mark>董天泽 物理学院</mark>



> **说明：**
>
> 1. **惊蛰⽉考**：<mark>未参加</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
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

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：



代码：

```python
while True:
    try:
        noun = True
        str0 = input()
        num = str0.count('@')
        if str0[0] == '@' or str0[0] == '.' or str0[-1] == '@' or str0[-1] == '.':
            noun = False
        if num == 1:
            num1 = str0.index('@')
            str1 = str0[num1:]
            num2 = str1.count('.')
            if num2 == 0:
                noun = False
            if str1[1] == '.':
                noun = False
            if str0[num1 - 1] == '.':
                noun = False

        else:
            noun = False

        if noun:
            print('YES')
        else:
            print('NO')
    except:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![0e736f1b2a6ea22265e4acd99378321](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\0e736f1b2a6ea22265e4acd99378321.png)



### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：



代码：

```python
n = int(input())
str0 = input()
list0 = ['' for i in range(n)]
list1 = []
num0 = 0
for i in range(1, len(str0) + 1):
    if i % n == 0:
        str1 = str0[i - n:i]
        num0 += 1
        str2 = ''
        if num0 % 2 == 0:
            str2 = str1[::-1]

        else:
            str2 = str1

        for j in range(n):
            list0[j] = list0[j] + str2[j]


print(''.join(list0))


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![114d24ec363d40edd1718a0433b17dd](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\114d24ec363d40edd1718a0433b17dd.png)



### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：



代码：

```python
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    dict0 = {}
    for i in range(n):
        list0 = [int(x) for x in input().split()]
        for x in list0:
            if x not in dict0:
                dict0[x] = 1
            else:
                dict0[x] += 1
    list0 = list(dict0.values())
    list0.sort()
    num0 = list0[-2]
    list1 = []
    for i in dict0:
        if dict0[i] == num0:
            list1.append(i)
    list1.sort()
    print(' '.join([str(x) for x in list1]))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![6e2bca2493f64e8f3b93009d2f65b6f](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\6e2bca2493f64e8f3b93009d2f65b6f.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：



代码：

```python
d = int(input())
n = int(input())
list0 = []
for i in range(1025):
    list1 = []
    for j in range(1025):
        list1.append(0)
    list0.append(list1)
for i in range(n):
    x, y, m = map(int, input().split())
    r1 = x - d
    r2 = x + d
    l1 = y - d
    l2 = y + d
    true_r1 = max(r1, 0)
    true_r2 = min(r2, 1024)
    true_l1 = max(l1, 0)
    true_l2 = min(l2, 1024)
    for k in range(true_r1, true_r2 + 1):
        for p in range(true_l1, true_l2 + 1):
            list0[k][p] += m
num0 = 0
max0 = 0
for k in range(1025):
    for p in range(1025):
        if list0[k][p] < max0:
            pass
        elif list0[k][p] > max0:
            num0 = 1
            max0 = list0[k][p]
        else:
            num0 += 1
print(num0, max0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![14fbd6177664f3b520c046db329ddef](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\14fbd6177664f3b520c046db329ddef.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：



代码：

```python
def inside(a, b, x, y):
    return 0 <= x < a and 0 <= y < b
def dfs(a, b, x_, y_, path, depth):
    if depth == a * b:
        return path

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

    for dx, dy in directions:
        x, y = x_ + dx, y_ + dy
        if inside(a, b, x, y) and (x, y) not in path:
            path.append((x, y))
            ans = dfs(a, b, x, y, path, depth + 1)
            if ans:
                return ans
            path.pop()

    return False
def trans_to_ans(ans):
    if ans:
        name_x = "ABCDEFGH"
        name_y = "12345678"
        a = ""
        for x, y in ans:
            a += name_x[x] + name_y[y]
        return a
    return "impossible"

t = int(input())
for _ in range(t):
    b, a = map(int, input().split())
    print(f"Scenario #{_ + 1}:")
    print(trans_to_ans(dfs(a, b, 0, 0, [(0, 0)], 1)))
    if _ != t - 1:
        print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![ea753c24974ffdfce7d47966a1a8a3c](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\ea753c24974ffdfce7d47966a1a8a3c.png)



### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：



代码：

```python
import heapq
def merge_min(s, b, n):
    visited = set()
    heap = []
    heapq.heappush(heap, (s[0] + b[0], 0, 0))
    visited.add((0, 0))
    res = []
    while len(res) < n and heap:
        sum_val, i, j = heapq.heappop(heap)
        res.append(sum_val)
        if i + 1 < len(s) and (i+1, j) not in visited:
            new_sum = s[i+1] + b[j]
            heapq.heappush(heap, (new_sum, i+1, j))
            visited.add((i+1, j))
        if j + 1 < len(b) and (i, j+1) not in visited:
            new_sum = s[i] + b[j+1]
            heapq.heappush(heap, (new_sum, i, j+1))
            visited.add((i, j+1))
    return res

T = int(input())
for _ in range(T):
    m, n = map(int, input().split())
    arrays = []
    for _ in range(m):
        row = list(map(int, input().split()))
        row.sort()
        arrays.append(row)
    current = arrays[0].copy()
    for i in range(1, m):
        b = arrays[i]
        new_current = merge_min(current, b, n)
        current = new_current
    print(' '.join(map(str, current)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![41332c8c31c4dc00d8f7bf6208ebbcf](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\41332c8c31c4dc00d8f7bf6208ebbcf.png)



## 2. 学习总结和收获

这次由于有其他课程与考试冲突了，就没去。自己回去做了一下，按考试算的话应该是AC4，第五题一开始没看明白什么意思（英语不太好），后面看了眼答案就明白要干啥了。最后一题直接交给Deepseek了。准备抽空做一做每日选做。











