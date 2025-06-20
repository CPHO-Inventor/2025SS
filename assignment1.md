# Assignment #1: 虚拟机，Shell & 大语言模型

Updated 2309 GMT+8 Feb 20, 2025

2025 spring, Complied by <mark>董天泽 物理学院</mark>



**作业的各项评分细则及对应的得分**

| 标准                                 | 等级                                                         | 得分 |
| ------------------------------------ | ------------------------------------------------------------ | ---- |
| 按时提交                             | 完全按时提交：1分<br/>提交有请假说明：0.5分<br/>未提交：0分  | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于2个：0分 | 1 分 |
| AC代码截图                           | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于：0分 | 1 分 |
| 清晰头像、PDF文件、MD/DOC附件        | 包含清晰的Canvas头像、PDF文件以及MD或DOC格式的附件：1分<br/>缺少上述三项中的任意一项：0.5分<br/>缺失两项或以上：0分 | 1 分 |
| 学习总结和个人收获                   | 提交了学习总结和个人收获：1分<br/>未提交学习总结或内容不详：0分 | 1 分 |
| 总得分： 5                           | 总分满分：5分                                                |      |
>
> 
>
> **说明：**
>
> 1. **解题与记录：**
>       - 对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>    
>2. **课程平台与提交安排：**
> 
>   - 我们的课程网站位于Canvas平台（https://pku.instructure.com ）。该平台将在第2周选课结束后正式启用。在平台启用前，请先完成作业并将作业妥善保存。待Canvas平台激活后，再上传你的作业。
> 
>       - 提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>3. **延迟提交：**
> 
>   - 如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
> 
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：



代码：

```python
list0 = [int(x) for x in input().split()]
a = list0[0] * list0[3] + list0[1] * list0[2]
b = list0[1] * list0[3]
m = min(a, b)
list1 = [2, 3]
for i in range(4, m + 1):
    noun = True
    for j in range(len(list1)):
        if i % list1[j] == 0:
            noun = False
    if noun:
        list1.append(i)
num = 0
while num < len(list1):
    if a % list1[num] == 0 and b % list1[num] == 0:
        a = a // list1[num]
        b = b // list1[num]
    else:
        num += 1
if b != 1:
    print(f'{a}/{b}')
else:
    print(a)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![bc4218c75ee3799d485a95c6eee1f86](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\bc4218c75ee3799d485a95c6eee1f86.png)



### 1760.袋子里最少数目的球

 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/




思路：



代码：

```python
class Solution(object):
    def minimumSize(self, nums, maxOperations):
        left, right, ans = 1, max(nums), 0
        while left <= right:
            y = (left + right) // 2
            ops = sum((x - 1) // y for x in nums)
            if ops <= maxOperations:
                ans = y
                right = y - 1
            else:
                left = y + 1
        
        return ans


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![f957855395dfb287611e7b0d739135d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\f957855395dfb287611e7b0d739135d.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135



思路：



代码：

```python
n, m = map(int, input().split())
list0 = [int(input()) for i in range(n)]
def find_num(y):
    step, num = 0, 1
    for i in list0:
        step += i
        if step > y:
            step = i
            num += 1
    return num

left, right = max(list0), sum(list0)
while left < right:
    mid = (left + right) // 2
    if find_num(mid) <= m:
        right = mid
    else:
        left = mid + 1

print(left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![61c21abe04e28930103aaa6cc9feb78](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\61c21abe04e28930103aaa6cc9feb78.png)



### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/



思路：



代码：

```python
from collections import defaultdict

n = int(input())
d = defaultdict(list)
for i in range(n):
    name, num = input().split('-')
    if num[-1] == 'M':
        d[name].append((num, float(num[:-1])/1000))
    else:
        d[name].append((num, float(num[:-1])))

sd = sorted(d)
for k in sd:
    nums = sorted(d[k],key=lambda x: x[1])
    value = ', '.join([i[0] for i in nums])
    print(f'{k}: {value}')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![c1a47b14bdb952745e677e4a63c4f8e](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\c1a47b14bdb952745e677e4a63c4f8e.png)



### Q5. 大语言模型（LLM）部署与测试

本任务旨在本地环境或通过云虚拟机（如 https://clab.pku.edu.cn/ 提供的资源）部署大语言模型（LLM）并进行测试。用户界面方面，可以选择使用图形界面工具如 https://lmstudio.ai 或命令行界面如 https://www.ollama.com 来完成部署工作。

测试内容包括选择若干编程题目，确保这些题目能够在所部署的LLM上得到正确解答，并通过所有相关的测试用例（即状态为Accepted）。选题应来源于在线判题平台，例如 OpenJudge、Codeforces、LeetCode 或洛谷等，同时需注意避免与已找到的AI接受题目重复。已有的AI接受题目列表可参考以下链接：
https://github.com/GMyhf/2025spring-cs201/blob/main/AI_accepted_locally.md

请提供你的最新进展情况，包括任何关键步骤的截图以及遇到的问题和解决方案。这将有助于全面了解项目的推进状态，并为进一步的工作提供参考。

#### 02808:校门外的树

![849770ed4b7e4d5a2fe9770c892242c](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\849770ed4b7e4d5a2fe9770c892242c.png)

![2afd68071c20e9cedb0a104c71349ac](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\2afd68071c20e9cedb0a104c71349ac.png)

![f582a59834fa85fc27190863839f1cc](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\f582a59834fa85fc27190863839f1cc.png)

#### 02524:宗教信仰

![a909e38f4c7ef3c65372b814c9c50ec](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a909e38f4c7ef3c65372b814c9c50ec.png)

![3ee49c24999e316cdc6faa363262851](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\3ee49c24999e316cdc6faa363262851.png)

![50608e10ff6a39f9c73f83d6d1da86f](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\50608e10ff6a39f9c73f83d6d1da86f.png)

### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

作者：Sebastian Raschka

请整理你的学习笔记。这应该包括但不限于对第一章核心概念的理解、重要术语的解释、你认为特别有趣或具有挑战性的内容，以及任何你可能有的疑问或反思。通过这种方式，不仅能巩固你自己的学习成果，也能帮助他人更好地理解这一部分内容。

预训练：提供了一种有效的方法，使得机器学习模型能够在缺乏大量标注数据的情况下，仍然能够高效地学习并应用于各种实际问题中。预训练后的模型可以作为基础模型用于后续的调控。



Transform架构及自注意力机制：

Transformer架构主要由编码器和解码器两大部分组成。两者都采用了自注意力机制来帮助模型理解输入序列中的每个元素与其他所有元素之间的关系，而无需受限于它们在序列中的位置。

自注意力机制：自注意力机制允许模型直接访问输入序列中任何位置的信息，从而有效捕捉长距离依赖关系。通过这种方式，Transformer架构能够生成更加连贯且上下文相关的文本。



## 2. 学习总结和个人收获

感觉题目不是特别难，就是好久没做题有点不熟练，在虚拟机上花了一些时间。开学初的几周并没有想象中的轻松，反而有一堆小事，因此花在数算的时间有限，准备之后抽时间做点题。





