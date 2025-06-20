# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

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

### LC222.完全二叉树的节点个数

bfs, dfs, binary + greedy,  https://leetcode.cn/problems/count-complete-tree-nodes/

如果用bfs写是简单级别，其他方法是中级难度。

思路：



代码：

```python
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
      
        level = 1
        queue = [root]
        root.val = 1
        while queue:
            tmp = []
            for node in queue:
                if node.left and not node.right:
                    return node.val * 2
                elif not node.left:
                    return node.val * 2 -1
                else:
                    node.left.val = node.val * 2
                    node.right.val = node.val * 2 + 1
                    tmp.append(node.left)
                    tmp.append(node.right)
            queue = tmp
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422221223594](C:\Users\18963\AppData\Roaming\Typora\typora-user-images\image-20250422221223594.png)





### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：



代码：

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        res, deque = [], collections.deque([root])
        while deque:
            tmp = collections.deque()
            for _ in range(len(deque)):
                node = deque.popleft()
                if len(res) % 2 == 0: tmp.append(node.val) 
                else: tmp.appendleft(node.val) 
                if node.left: deque.append(node.left)
                if node.right: deque.append(node.right)
            res.append(list(tmp))
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![5b437d690fccd301e9f0b31a74edf2a](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\5b437d690fccd301e9f0b31a74edf2a.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：



代码：

```python
import heapq
class Node():
    def __init__(self,weight):
        self.left=None
        self.right=None
        self.weight=weight
    def  __lt__(self,other):
        return self.weight<other.weight
def buildtree(w):
    h=[Node(i) for i in w]
    heapq.heapify(h)
    while len(h)>1:
        x=heapq.heappop(h)
        y=heapq.heappop(h)
        he=Node(x.weight+y.weight)
        he.left=x
        he.right=y
        heapq.heappush(h,he)
    return h[0]
def epl(root,depth):
    if not root:
        return 0
    if not root.left and not root.right:
        return depth*root.weight
    return epl(root.left,depth+1)+epl(root.right,depth+1)
input()
w=list(map(int,input().split()))
root=buildtree(w)
print(epl(root,0))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![0b0e1c6eb20be84fc9fd145cf265d2d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\0b0e1c6eb20be84fc9fd145cf265d2d.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：



代码：

```python
from collections import deque
class Node():
    def __init__(self,val):
        self.left=None
        self.right=None
        self.val=val
def insert(root,node):
    if not root:
        return node
    if root.val>node.val:
        root.left=insert(root.left,node)
    else:
        root.right=insert(root.right,node)
    return root
def level_order(root):
    q=deque([root])
    ans=[]
    while q:
        node=q.popleft()
        ans.append(str(node.val))
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return ans
l=list(map(int,input().split()))
root=None
visit=set()
for i in l:
    if i not in visit:
        visit.add(i)
        root=insert(root,Node(i))
print(' '.join(level_order(root)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![c79095f788deac937ff5125756aea2a](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\c79095f788deac937ff5125756aea2a.png)



### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：



代码：

```python
import heapq

def main():
    # 初始化空堆
    heap = []
    
    # 输入操作次数
    n = int(input())
    
    for _ in range(n):
        line = input().split()
        type_op = int(line[0])
        
        if type_op == 1:  # 添加操作
            u = int(line[1])
            heapq.heappush(heap, u)  # 将元素插入堆中
        
        elif type_op == 2:  # 输出并删除最小元素操作
            if heap:
                print(heapq.heappop(heap))  # 弹出并输出堆顶元素（最小值）
            else:
                print("Heap is empty!")  # 如果堆为空，提示错误信息

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![7cf970d75302e155e39f6a4fc59b85a](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\7cf970d75302e155e39f6a4fc59b85a.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：



代码：

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        if node.char:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        if node.char:
            decoded += node.char
            node = root
    return decoded

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

#string = input().strip()
#encoded_string = input().strip()

# 构建哈夫曼编码树
huffman_tree = build_huffman_tree(characters)

# 编码和解码
codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        if line:
            strings.append(line)
        else:
            break
    except EOFError:
        break

results = []
#print(strings)
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![c23a573e70e9f1689c1e0bc49a3401b](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\c23a573e70e9f1689c1e0bc49a3401b.png)



## 2. 学习总结和收获

感觉有点难度，这周身体不太舒服，最后两题直接看答案了。还在补期中落下的部分。











