# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

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

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：



代码：

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


n, k = map(int, input().split())


nodes = [Node(i) for i in range(1, n + 1)]
for i in range(n):
    nodes[i].next = nodes[(i + 1) % n]

prev = nodes[-1]
current = nodes[0]
result = []

while current.next != current:

    for _ in range(k - 1):
        prev = current
        current = current.next

    result.append(str(current.data))
    prev.next = current.next
    current = current.next

print(' '.join(result))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408171343224](C:\Users\18963\AppData\Roaming\Typora\typora-user-images\image-20250408171343224.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：



代码：

```python
n, k = map(int, input().split())
length = [int(input()) for _ in range(n)]

def can_find(mid, k):
    if mid == 0:
        return False

    count = 0
    for len in length:
        count += len // mid
    return count >= k

l = 0
r = max(length)
result = 0

while l <= r:
    mid = (l + r) // 2
    if can_find(mid, k):
        result = mid
        l = mid + 1
    else:
        r = mid - 1

print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![56c64c7c3ad9f782a62c1b90b67f97a](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\56c64c7c3ad9f782a62c1b90b67f97a.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：



代码：

```python
from collections import deque
import sys
class TreeNode:
    def __init__(self, name, degree):
        self.name = name
        self.degree = degree
        self.children = []

def main():
    n = int(sys.stdin.readline())
    result = []

    for _ in range(n):
        line = sys.stdin.readline().strip().split()
        nodes = []
        for i in range(0, len(line), 2):
            name = line[i]
            degree = int(line[i + 1])
            nodes.append(TreeNode(name, degree))

        if not nodes:
            continue

        q = deque([nodes[0]])
        j = 1
        while q:
            current_node = q.popleft()
            d = current_node.degree
            if d > 0:
                children = nodes[j:j + d]
                current_node.children = children
                q.extend(children)
                j += d

        post_order_list = []
        def post_order(node):
            for child in node.children:
                post_order(child)
            post_order_list.append(node.name)

        post_order(nodes[0])
        result.extend(post_order_list)

    print(' '.join(result))

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![0b6cedc960b4f9bf79253d77f99d3b4](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\0b6cedc960b4f9bf79253d77f99d3b4.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：



代码：

```python
t = int(input())
nums = sorted(list(map(int, input().split())))
l = 0; r = len(nums) - 1

ans = float("inf")
while l != r:
    cur = nums[l] + nums[r]
    if cur < t:
        l += 1
    else:
        r -= 1

    if abs(cur - t) <= abs(ans - t):
        if abs(cur - t) < abs(ans - t):
            ans = cur
        else:
            ans = min(ans, cur)

print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![48f2853e199bfdf301f59a6d243af30](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\48f2853e199bfdf301f59a6d243af30.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：这也能过？



代码：

```python
import bisect
Lis = [11,31,41,61,71,101,131,151,181,191,211,241,251,271,281,311,331,401,421,431,461,491,521,541,571,601,631,641,661,691,701,751,761,811,821,881,911,941,971,991,1021,1031,1051,1061,1091,1151,1171,1181,1201,1231,1291,1301,1321,1361,1381,1451,1471,1481,1511,1531,1571,1601,1621,1721,1741,1801,1811,1831,1861,1871,1901,1931,1951,2011,2081,2111,2131,2141,2161,2221,2251,2281,2311,2341,2351,2371,2381,2411,2441,2521,2531,2551,2591,2621,2671,2711,2731,2741,2791,2801,2851,2861,2971,3001,3011,3041,3061,3121,3181,3191,3221,3251,3271,3301,3331,3361,3371,3391,3461,3491,3511,3541,3571,3581,3631,3671,3691,3701,3761,3821,3851,3881,3911,3931,4001,4021,4051,4091,4111,4201,4211,4231,4241,4261,4271,4391,4421,4441,4451,4481,4561,4591,4621,4651,4691,4721,4751,4801,4831,4861,4871,4931,4951,5011,5021,5051,5081,5101,5171,5231,5261,5281,5351,5381,5431,5441,5471,5501,5521,5531,5581,5591,5641,5651,5701,5711,5741,5791,5801,5821,5851,5861,5881,5981,6011,6091,6101,6121,6131,6151,6211,6221,6271,6301,6311,6361,6421,6451,6481,6491,6521,6551,6571,6581,6661,6691,6701,6761,6781,6791,6841,6871,6911,6961,6971,6991,7001,7121,7151,7211,7321,7331,7351,7411,7451,7481,7541,7561,7591,7621,7681,7691,7741,7841,7901,7951,8011,8081,8101,8111,8161,8171,8191,8221,8231,8291,8311,8431,8461,8501,8521,8581,8641,8681,8731,8741,8761,8821,8831,8861,8941,8951,8971,9001,9011,9041,9091,9151,9161,9181,9221,9241,9281,9311,9341,9371,9391,9421,9431,9461,9491,9511,9521,9551,9601,9631,9661,9721,9781,9791,9811,9851,9871,9901,9931,9941]
t = int(input())
for _ in range(1, t + 1):
    print(f"Case{_}:")

    n = int(input())
    cut = bisect.bisect_left(Lis, n)
    if cut:
        print(" ".join(list(map(str, Lis[:cut]))))
    else:
        print("NULL")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![3e964a6160e45519bb84428fe3ddc75](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\3e964a6160e45519bb84428fe3ddc75.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：



代码：

```python
import sys
from collections import defaultdict

def process_icpc_results(m, submissions):
    teams = defaultdict(lambda: {'solved': set(), 'attempts': defaultdict(int), 'total_attempts': 0})
    
    for record in submissions:
        team_name, problem, result = record.split(',')
        team_name = team_name.strip()
        problem = problem.strip()
        result = result.strip()
        
        teams[team_name]['attempts'][problem] += 1
        teams[team_name]['total_attempts'] += 1
        
        if result == 'yes':
            teams[team_name]['solved'].add(problem)
    
    ranking = sorted(
        teams.items(),
        key=lambda item: (-len(item[1]['solved']), item[1]['total_attempts'], item[0])
    )
    
    for rank, (team_name, data) in enumerate(ranking[:12], start=1):
        print(rank, team_name, len(data['solved']), data['total_attempts'])

if __name__ == "__main__":
    m = int(sys.stdin.readline().strip())
    submissions = [sys.stdin.readline().strip() for _ in range(m)]
    process_icpc_results(m, submissions)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![6111ff47ab7242fe43bb7aa08fe2484](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\6111ff47ab7242fe43bb7aa08fe2484.png)



## 2. 学习总结和收获

期中要复习的太多了就没去机考。但想来上次机考没去，这次也没去，只能说下次一定去了（

也没有按照考试的时间做，就自己简单做了下，感觉后面两题特别麻烦，整体而言感觉一般，期中之后需要多练了。



