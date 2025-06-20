## cheat sheet

### 1. oop

在Python中定义新类的做法是，提供一个类名以及一整套与函数定义语法类似的方法定义。以下是一个方法定义框架

	class Fraction
	#the methods go here




所有类都应该首先提供构造方法。构造方法定义了数据对象的创建方式。要创建一个Fraction对象，需要提供分子和分母两部分数据。在Python中，构造方法总是命名为`__init__`（即在init的前后分别有两个下划线）。

代码 Fraction类及其构造方法

```
def __init__(self, top, bottom):
    
    self.num = top
    self.den = bottom
```

### 2. 二分查找

```
# 返回 x 在 arr 中的索引，如果不存在返回 -1
def binarySearch(arr, l, r, x):
    # 基本判断
    if r >= l:

        mid = int(l + (r - l) / 2)

        # 元素整好的中间位置
        if arr[mid] == x:
            return mid

            # 元素小于中间位置的元素，只需要再比较左边的元素
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)

            # 元素大于中间位置的元素，只需要再比较右边的元素
        else:
            return binarySearch(arr, mid + 1, r, x)

    else:
        # 不存在
        return -1
```

例子[OpenJudge - 22528:厚道的调分方法](http://cs101.openjudge.cn/practice/22528/)

```
grade = [float(x) for x in input().split()]
grade.sort()
num = int(len(grade) * 0.4)
line0 = grade[num]
left = 0
right = 1000000000
ans = 0
while right >= left:
    mid = int((right + left) / 2)
    gd = line0 * mid / 1000000000 + 1.1 ** (line0 * mid / 1000000000)
    if gd >= 85:
        ans = mid
        right = mid - 1
    else:
        left = mid + 1
print(ans)
```

### 3.tree

#### 3.1 树的基础

#### 3.1.1 树的建立

##### class法

```python
class TreeNode:
	def __init__(self, value):
        # 二叉树（binary tree)
        self.value = value
		self.left = None
		self.right = None
        
        # 多叉树（N-nary tree)
        self.value = value
        self.children = []
        
        # 左儿子右兄弟树（First child / Next sibling representation）
		self.value = value
		self.firstChild = None
		self.nextSibling = None
		# 这玩意像个链表，有时候会很好用

n = int(input())
# 一般而言会有一个存Nodes的dict或是list
nodes = [TreeNode() for i in range(n)]
# 甚至会让你找root，这也可以用于记录森林的树量
has_parents = [False] * n

for i in range(n):
    opt = map(int, input().spilt())
    if opt[0] != -1:
        nodes[i].left = nodes[opt[0]]
        has_parent[opt[0]] = True
    if opt[1] != -1:
        nodes[i].right = nodes[opt[1]]
        has_parent[opt[1]] = True
# 这里完成了树的建立

root = has_parent.index(False) # 对于一棵树而言, root可以被方便的确定
```

#### 3.1.2  遍历

##### 前序遍历（Pre-order Traversal）

``` 
def preorder_traversal(root):
    if root:
        print(root.val)  # 访问根节点
        preorder_traversal(root.left)  # 递归遍历左子树
        preorder_traversal(root.right)  # 递归遍历右子树
```

##### 中序遍历（In-order Traversal）

```
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)  # 递归遍历左子树
        print(root.val)  # 访问根节点
        inorder_traversal(root.right)  # 递归遍历右子树
```

##### 后序遍历（Post-order Traversal）

```
def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)  # 递归遍历左子树
        postorder_traversal(root.right)  # 递归遍历右子树
        print(root.val)  # 访问根节点
```

##### 层序遍历：按层从左到右依次遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

```
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level)

        return result
```

#### 3.1.3 小模型

#### 1.求二叉树高度

###### 练习M27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def length_path(node):
    if not node:
        return -1
    leftnum = length_path(node.left)
    rightnum = length_path(node.right)
    return max(leftnum, rightnum) + 1

def count_leaves(node):
    if not node:
        return 0
    if node.left == None and node.right == None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


n = int(input())
list0 = [False] * n
nodes = [TreeNode() for _ in range(n)]

for i in range(n):
    lcld, rcld = map(int, input().split())
    if lcld != -1:
        nodes[i].left = nodes[lcld]
        list0[lcld] = True
    if rcld != -1:
        nodes[i].right = nodes[rcld]
        list0[rcld] = True

rootindex = list0.index(False)
root = nodes[rootindex]

print(length_path(root), count_leaves(root))
```

#### 2. 判断两棵树是否相同

```
def is_same_tree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))
```

例子

https://leetcode.cn/problems/symmetric-tree/

```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        def isMirror(left: TreeNode, right: TreeNode) -> bool:
            if not left and not right:
                return True
            if not left or not right:
                return False
            return (left.val == right.val) and isMirror(left.left, right.right) and isMirror(left.right, right.left)

        return isMirror(root.left, root.right)
```

#### 3. 寻找二叉搜索树中的最小值/最大值

搜索二叉树默认左侧比右侧小，找最小往左侧找，找最大往右侧找

```
def find_min(root):
    if not root.left:
        return root.val
    return find_min(root.left)

def find_max(root):
    if not root.right:
        return root.val
    return find_max(root.right)
```

https://leetcode.cn/problems/kth-smallest-element-in-a-bst/

```
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []
        def pre_order(node):
            if not node or len(res) == k:
                return
            pre_order(node.left)
            if len(res) == k:
                return
            res.append(node.val)
            if len(res) == k:
                return
            pre_order(node.right)
        
        pre_order(root)
        return res[-1]
```

#### 4. 判断是否为平衡二叉树

先正常写出计算高度的代码，再添加判断是否为平衡树的代码

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def if_balance(root):
    def check_height(node):
        if not node:
            return 0

        lefttree = check_height(node.left)
        if lefttree == -1:
            return -1
        righttree = check_height(node.right)
        if righttree == -1:
            return -1

        if abs(righttree - lefttree) > 1:
            return -1

        return max(check_height(node.left), check_height(node.right)) + 1

    return check_height(root) != -1
```

#### 3.2 树的转化

$$
括号嵌套树 \Rightarrow 正常的多叉树
$$

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        
def build_Tree(string):
    node = None
    stack = [] # 及时处理
    for chr in string:
    	if chr.isalpha(): # 这个是一个判断函数，多见于buffer
            node = TreeNode(chr)
            if stack:
                stack[-1].children.append(node)
        elif chr == "(":
            stack.append(node)
            node = None # 及时更新
        elif chr == ")":
            node = stack.pop() # 最后返回树根
        else:
            continue
    return node
# stack在这里的运用非常符合栈的定义和特征

def preorder(root):
    output = [root.val]
    for i in root.children: # 这里的输出不一样，因为孩子不止一个
        output.extend(preorder(i))
    return "".join(output)
```

$$
括号嵌套树 \Leftarrow 正常的多叉树
$$

```python
def convert_to_bracket_tree(node):
    # 两个终止条件
    if not node:
        return ""
    if not node.children:
        return node.val
    
    result = node.val + "("
    for i, child in enumerate(node.children):
        result += convert_to_bracket_tree(child)
        if i != len(node.children) - 1:
            result += "," # 核心是“，”的加入，这里选择在一层结束前加入
    result += ")"
    
    return result
```

$$
文件转化树
$$

```python
class Dir:
    def __init__(self, file_name):
        self.name = file_name
        self.dirs = []
        self.files = []

    def show(self, dir_name, layers = 0): # 这里把layer作为遍历的“线”
        layer = layers
        result = ["|     " * layer + dir_name.name]
        dir_name.files.sort()

        for dir in dir_name.dirs:
            result.extend(self.show(dir, layer + 1))
        for file in dir_name.files:
            result.extend(["|     " * layer + file]) # extend(str)会把字符串拆开
        return result


n = 0
while True:
    n += 1
    stack = [Dir("ROOT")] # 这的输入比较难，其实也是采用的是栈的思路——及时处理，及时退出
    while (s := input()) != "*":
        if s == "#":
            exit()
        if s[0] == "f":
            stack[-1].files.append(s)
        elif s[0] == "d":
            stack.append(Dir(s))
            stack[-2].dirs.append(stack[-1])
        else:
            stack.pop()

    print(f"DATA SET {n}:")
    print(*stack[0].show(stack[0]), sep="\n") # result是个列表，存储字符串
    print() # 分割线
```

$$
建立起表达式树，按层次遍历表达式树的结果前后颠倒就得到队列表达式
$$

#### 3.3 各种树

#### 3.3.1 哈夫曼算法

- **前缀码**：哈夫曼编码使用的是前缀码技术，即任何一个字符的编码都不是其他字符编码的前缀。这保证了编码的唯一可解性。

- **频率统计**：对需要编码的数据中的每个符号进行频率统计。

- **构建哈夫曼树**：基于符号出现的频率来构建一棵二叉树（哈夫曼树）。频率越高的符号距离树根越近，从而获得更短的编码。

  要构建一个最优的哈夫曼编码树，首先需要对给定的字符及其权值进行排序。然后，通过重复合并权值

  最小的两个节点（或子树），直到所有节点都合并为一棵树为止。

  下面是用 Python 实现的代码：

  ```
  import heapq
  
  class Node:
      def __init__(self, char, freq):
          self.char = char
          self.freq = freq
          self.left = None
          self.right = None
  
      def __lt__(self, other):
          return self.freq < other.freq
  
  def huffman_encoding(char_freq):
      heap = [Node(char, freq) for char, freq in char_freq.items()]
      heapq.heapify(heap)
  
      while len(heap) > 1:
          left = heapq.heappop(heap)
          right = heapq.heappop(heap)
          merged = Node(None, left.freq + right.freq) # note: 合并之后 char 字典是空
          merged.left = left
          merged.right = right
          heapq.heappush(heap, merged)
  
      return heap[0]
  
  # 同样的 以depth作为递归深度的线
  def external_path_length(node, depth=0):
      if node is None:
          return 0
      if node.left is None and node.right is None:
          return depth * node.freq
      return (external_path_length(node.left, depth + 1) +
              external_path_length(node.right, depth + 1))
  
  def main():
      char_freq = {'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 8, 'f': 9, 'g': 11, 'h': 12}
      huffman_tree = huffman_encoding(char_freq)
      external_length = external_path_length(huffman_tree)
      print("The weighted external path length of the Huffman tree is:", external_length)
  
  if __name__ == "__main__":
      main()
  ```

#### 3.3.2 二叉搜索树

二叉搜索树依赖于这样一个性质：<mark>小于父节点的键都在左子树中，大于父节点的键则都在右子树中。我们称这个性质为二叉搜索性。</mark>

http://cs101.openjudge.cn/practice/22275/

```
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def build(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = Node(root_val)
    root_index = inorder.index(root_val)
    root.left = build(preorder[1:root_index + 1], inorder[:root_index])
    root.right = build(preorder[root_index + 1:], inorder[root_index + 1:])
    return root


def postorder(root):
    if not root:
        return []
    if root.left is None and root.right is None:
        return [root.val]
    result = []
    result += postorder(root.left)
    result += postorder(root.right)
    result += [root.val]
    return result


input()
preorder = list(map(int, input().split()))
inorder = sorted(preorder)
root = build(preorder, inorder)
result = postorder(root)
print(' '.join(map(str, result)))
```

二叉搜索树快排

```
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root

def inorder_traversal(root, result):
    if root:
        inorder_traversal(root.left, result)
        result.append(root.val)
        inorder_traversal(root.right, result)

def quicksort(nums):
    if not nums:
        return []
    root = TreeNode(nums[0])
    for num in nums[1:]:
        insert(root, num)
    result = []
    inorder_traversal(root, result)
    return result

# 示例输入
nums = [5, 2, 7, 1, 8, 3]

# 使用二叉搜索树实现快速排序
sorted_nums = quicksort(nums)

# 打印排序结果
print(sorted_nums)
```

#### 3.3.3 **邻接表的BFS和DFS运用**

**Source Code: BFS in Python**

```
graph = {'A': ['B', 'C', 'E'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F', 'G'],
         'D': ['B'],
         'E': ['A', 'B', 'D'],
         'F': ['C'],
         'G': ['C']}


def bfs(graph, initial):
    visited = []
    queue = [initial]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]

            for neighbour in neighbours:
                queue.append(neighbour)
    return visited


print(bfs(graph, 'A'))
```

**Source Code: DFS in Python**

[OpenJudge - 28013:堆路径](http://cs101.openjudge.cn/practice/28013/)

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.value = val
        self.left = left
        self.right = right

def dfs(root, path):
    if not root.left and not root.right:
        paths.append(path + [root.value])
        return

    if root.right:
        dfs(root.right, path + [root.value])
    if root.left:
        dfs(root.left, path + [root.value])

def judge(nums):
    incre = decre = True
    for i in range(1, len(nums)):
        if nums[i - 1] < nums[i]:
            decre = False
        if nums[i - 1] > nums[i]:
            incre = False

    if incre:
        return 1
    elif decre:
        return -1
    else:
        return 0

paths = []
n = int(input())
list0 = [int(x) for x in input().split()]
list_node = [TreeNode(list0[i]) for i in range(n)]
for i in range(n):
    if 2 * i + 1 < n:
        list_node[i].left = list_node[2 * i + 1]
    if 2 * i + 2 < n:
        list_node[i].right = list_node[2 * i + 2]

dfs(list_node[0], [])

for p in paths:
    print(' '.join([str(x) for x in p]))

if all(judge(_) == -1 for _ in paths):
    print("Max Heap")
elif all(judge(_) == 1 for _ in paths):
    print("Min Heap")
else:
    print("Not Heap")
```

## 4.一些杂的代码

### 4.1 cmp_to_key

```
from functools import cmp_to_key
def compar(a,b):
    if a>b:
        return 1#大的在后
    if a<b:
        return -1#小的在前
    else:
        return 0#返回零不变位置
l=[1,5,2,4,6,7,6]
l.sort(key=cmp_to_key(compar))
print(l)#[1,2,4,5,6,6,7]
```

### 4.2 保留小数

```
number = 3.14159
formatted_number = "{:.2f}".format(number)
print(formatted_number)  # 输出: 3.14
```

### 4.3 全排列

```
from itertools import permutations 
# Get all permutations of [1, 2, 3] 
perm = permutations([1, 2, 3]) 
# Get all permutations of length 2 
perm2 = permutations([1, 2, 3], 2) 
# Print the obtained permutations 
for i in list(perm): 
    print (i) 
```

binary search + greedy二分:

河中跳房子

```
L,n,m = map(int,input().split())
rock = [0]
for i in range(n):
    rock.append(int(input()))
rock.append(L)

def check(x):
    num = 0
    now = 0
    for i in range(1, n+2):
        if rock[i] - now < x:
            num += 1
        else:
            now = rock[i]
            
    if num > m:
        return True
    else:
        return False
lo, hi = 0, L+1
ans = -1
while lo < hi:
    mid = (lo + hi) // 2
    
    if check(mid):
        hi = mid
    else:               # 返回False，有可能是num==m
        ans = mid       # 如果num==m, mid就是答案
        lo = mid + 1
        
#print(lo-1)
print(ans)
```



### 4.4 单调栈

即人为控制栈内元素单调，找某侧最近⼀个⽐其⼤的值，使⽤单调栈维持栈内元素递减；找某侧最近⼀个⽐其⼩的值使⽤单调栈，维持栈内元素递增 ….

```
stack=[]
water=0
n=len(height)
for i in range(n):
	while stack and height[stack[-1]]<height[i]:
		top=stack.pop()
		if not stack:
			break
		d=i-stack[-1]-1
		h=min(height[stack[-1]],height[i])-height[top]
		water+=d*h
	stack.append(i)
return water
```



### 4.5 十大排序

包括：冒泡排序（Bubble Sort），插入排序（Insertion Sort），选择排序（Selection Sort），希尔排序（Shell Sort），归并排序（Merge Sort），快速排序（Quick Sort），堆排序（Heap Sort），计数排序（Counting Sort），桶排序（Bucket Sort），基数排序（Radix Sort）

#### 4.5.1 冒泡

```
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if (swapped == False):
            break
```

#### 4.5.2 选择排序

```
A = [64, 25, 12, 22, 11]
for i in range(len(A)):
 min_idx = i
 for j in range(i + 1, len(A)):
     if A[min_idx] > A[j]:
         min_idx = j
 A[i], A[min_idx] = A[min_idx], A[i]
print(' '.join(map(str, A)))
```

#### 4.5.3 快排

```
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)
def partition(arr, left, right):
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        while i <= right and arr[i] < pivot:
            i += 1
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i
arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)
```

### 4.6 递归优化

###### 1.增加递归深度限制

```
import sys
sys.setrecursionlimit(1 << 30) # 将递归深度限制设置为 2^30
```

###### 2.缓存中间结果

建个列表or字典调用或者直接内置函数缓存中间结果

```
from functools import lru_cache
@lru_cache(maxsize=None)
```

## 5.基础数据结构

### 5.1栈

#### 5.1.1 匹配括号

有多种括号：

```
def par_checker(symbol_string):
    s = [] # Stack()
    balanced = True
    index = 0 
    while index < len(symbol_string) and balanced:
        symbol = symbol_string[index] 
        if symbol in "([{":
            s.append(symbol) # push(symbol)
        else:
            top = s.pop()
            if not matches(top, symbol):
                balanced = False
        index += 1
        #if balanced and s.is_empty():
        if balanced and not s:
            return True 
        else:
            return False   
def matches(open, close):
    opens = "([{"
    closes = ")]}"
    return opens.index(open) == closes.index(close)
print(par_checker('{{}}[]]'))
```

例http://cs101.openjudge.cn/practice/03704括号匹配问题

```
lines = []
while True:
    try:
        lines.append(input())
    except EOFError:
        break
for s in lines:
    l=[]
    mark=[]
    for i in range(len(s)):
        if s[i]=='(':
            l.append(i)
            mark+=' '
        elif s[i]==')':
            if not l:
                mark+='?'
            else:
                l.pop()
                mark+=' '
        else:
            mark+=' '
    while l:
        ind=l.pop()
        mark[ind]='$'
    print(s)
    print(''.join(mark))
```

#### 5.1.2 中序、前序和后序表达式(含调度场算法)

例：中序表达式转后序表达式http://cs101.openjudge.cn/practice/24591/

```
n=int(input())
value={'(':1,'+':2,'-':2,'*':3,'/':3}
for _ in range(n):
    put=input()
    stack=[]
    out=[]
    number=''
    for s in put:
        if s.isnumeric() or s=='.':
            number+=s
        else:
            if number:
                num=float(number)
                out.append(int(num) if num.is_integer() else num)
                number=''
            if s=='(':
                stack.append(s)
            elif s==')':
                while stack and stack[-1]!='(':
                    out.append(stack.pop())
                stack.pop()
            else:
                while stack and value[stack[-1]]>=value[s]:
                    out.append(stack.pop())
                stack.append(s)
    if number:
        num = float(number)
        out.append(int(num) if num.is_integer() else num)
    while stack:
        out.append(stack.pop())
    print(*out,sep=' ')

```

#### 5.1.3 后序表达式求值

在读取后缀表达式的过程中，每当你遇到一个操作数，就将其压入栈中。一旦遇到操作符，就从栈中弹出适当数量的操作数（对于二元操作符来说是两个），执行相应的计算，并将结果压回到栈中。这个过程会一直持续到表达式的末尾，最终栈顶元素就是整个表达式的计算结果

例 后序表达式求值http://cs101.openjudge.cn/practice/24588/

```
def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2
for _ in range(int(input())):
    put=input().split()
    stack=[]
    for token in put:
        if token in '+-*/':
            a=stack.pop()
            b=stack.pop()
            stack.append(doMath(token,b,a))
        else:
            stack.append(float(token))
    print(f"{stack[-1]:.2f}")
```

### 1.2 图算法

#### 1.2.1基本图算法

##### 1.2.1.1 宽度优先搜索(bfs)(与计概相同)

http://cs101.openjudge.cn/practice/28046/ 词梯问题

```
from collections import defaultdict,deque
n=int(input())
words=[input() for i in range(n)]
graph=defaultdict(list)
tongs=defaultdict(set)
for word in words:
    for i in range(4):
        tong=f"{word[:i]}_{word[i+1:]}"
        tongs[tong].add(word)
for w in tongs.values():
    for word1 in w:
        for word2 in w-{word1}:
            graph[word1].append(word2)
queue=deque()
visit=set()
ws,we=input().split()
queue.append((ws,[ws]))
visit.add(ws)
while queue:
    word,path=queue.popleft()
    if word==we:
        print(' '.join(path))
        break
    for neibour in graph[word]:
        if neibour not in visit:
            visit.add(neibour)
            queue.append((neibour,path+[neibour]))
else:
    print('NO')
```

##### 1.2.1.2 深度优先搜索(dfs)(与计概相同)

递归实现的DFS算法的时间复杂度为O(V + E)，空间复杂度为O(V)

 http://cs101.openjudge.cn/practice/28050/ 骑士周游问题

```
from collections import defaultdict
d=[(2,-1),(2,1),(1,-2),(1,2),(-2,-1),(-2,1),(-1,-2),(-1,2)]
graph=defaultdict(list)
def trans(row,col):
    return row*n+col
def fneibour(row,col):
    neibour=[]
    for dx,dy in d:
        if 0<=row+dx<n and 0<=col+dy<n:
            neibour.append([row+dx,col+dy])
    return neibour
def cou(dot):
    c=0
    for nei in graph[dot]:
        if visit[nei]:
            c+=1
    return c
def sneibour(dot):
    neibour=[nei for nei in graph[dot] if visit[nei]]
    neibour.sort(key=lambda x:cou(x))
    return neibour
def dfs(count,dot):
    if count==n**2-1:
        return True
    visit[dot]=False
    neibour = sneibour(dot)
    for nei in neibour:
        if dfs(count+1,nei):
            return True
    else:
        visit[dot]=True
        return False
n=int(input())
sr,sc=map(int,input().split())
visit=[True]*(n**2)
for row in range(n):
    for col in range(n):
        for nr,nc in fneibour(row, col):
            graph[trans(row,col)].append(trans(nr,nc))

print(['fail','success'][dfs(0,trans(sr,sc))])
```

#### 1.2.2 拓展图算法

##### 1.2.2.1 拓扑排序

https://sunnywhy.com/sfbj/10/3/382 有向图判环

示例程序：

```
from collections import defaultdict,deque
n,m=map(int,input().split())
graph=defaultdict(list)
indegree=[0]*n
for i in range(m):
    vs,ve=map(int,input().split())
    graph[vs].append(ve)
    indegree[ve]+=1
l=[i for i in range(n) if indegree[i]==0]
quene=deque(l)
ans=[]
while quene:
    vert=quene.popleft()
    ans.append(quene)
    for nbr in graph[vert]:
        indegree[nbr]-=1
        if indegree[nbr]==0:
            quene.append(nbr)
print(['Yes','No'][len(ans)==n])
```

##### 1.2.2.2 强连通单元（SCCs）

Kosaraju算法的关键在于第二次DFS的顺序，它保证了在DFS的过程中，我们能够优先访问到整个图中的强连通分量。因此，Kosaraju算法的时间复杂度为O(V + E)，其中V是顶点数，E是边数。

以下是Kosaraju算法的Python实现：

```
from collections import defaultdict
def dfs1(vert):
    visit[vert]=True
    for nbr in graph[vert]:
        if not visit[nbr]:
            dfs1(nbr)
    stack.append(vert)
def Reverse(graph):
    graph2 = defaultdict(list)
    for vert in graph.keys():
        for nbr in graph[vert]:
            graph2[nbr].append(vert)
    return graph2 
def dfs2(vert):
    visit[vert] = True
    for nbr in graph2[vert]:
        if not visit[nbr]:
            dfs2(nbr)
    scc.append(vert)
graph=defaultdict(list)
n,m=map(int,input().split())
for i in range(m):
    vs,ve=map(int,input().split())
    graph[vs].append(ve)
stack=[]
visit=[False]*n
for vert in range(n):
    if not visit[vert]:
        dfs1(vert)
graph2=Reverse(graph)
visit=[False]*n
sccs=[]
while stack:
    vert=stack.pop()
    if not visit[vert]:
        scc=[]
        dfs2(vert)
        sccs.append(scc)
print(*sccs,sep='\n')
```

##### 1.2.2.3 最短路径

###### 1.dijkstra算法(计概学过)

非常重要的一点是，<mark>Dijkstra算法只适用于边的权重均为正的情况</mark>。如果图2中有一条边的权重为负，那么Dijkstra算法永远不会退出。

总的时间复杂度为$O((V+E) \log V)$

示例 https://leetcode.cn/problems/network-delay-time/description/网络延迟时间

```
from collections import defaultdict
import heapq
graph=defaultdict(dict)
for vs,ve,w in times:
    graph[vs][ve]=w
h=[(0,k)]
ltime=[0]+[20000]*n
ltime[k]=0
heapq.heapify(h)
while h:
    time,vert=heapq.heappop(h)
    if ltime[vert]<time:
        continue
    for nbr in graph[vert].keys():
        nt=time+graph[vert][nbr]
        if nt<ltime[nbr]:
            ltime[nbr]=nt
            heapq.heappush(h,(nt,nbr))
ans=max(ltime)
return ans if ans<20000 else -1

```

###### 2.多源最短路径Floyd-Warshall算法

求解所有顶点之间的最短路径可以使用**Floyd-Warshall算法**，它是一种多源最短路径算法。Floyd-Warshall算法可以在有向图或无向图中找到任意两个顶点之间的最短路径。

算法的基本思想是通过一个二维数组来存储任意两个顶点之间的最短距离。初始时，这个数组包含图中各个顶点之间的直接边的权重，对于不直接相连的顶点，权重为无穷大。然后，通过迭代更新这个数组，逐步求得所有顶点之间的最短路径。

思想：动态规划 + 三重循环

状态定义：`dist[i][j]` 表示 i 到 j 的最短路径长度

转移方程：

`dist[i][j]=min⁡(dist[i][j], dist[i][k]+dist[k][j])`

表示是否通过中间点 k 能让路径更短

最终得出任意两点之间的最短路径

用python实现

```
from collections import defaultdict
n,m=map(int,input().split())
graph=defaultdict(dict)
for _ in range(m):
    vs,ve,w=map(int,input().split())
    graph[vs][ve]=w
dist=[[float('inf')]*n for i in range(n)]
for i in range(n):
    dist[i][i]=0
    for j in graph[i].keys():
        dist[i][j]=graph[i][j]
print(*dist,sep='\n')
for k in range(n):
    for i in range(n):
        for j in range(n):
            dist[i][j]=min(dist[i][k]+dist[k][j],dist[i][j])
print(*dist,sep='\n')
```

需要求出路经时：

http://cs101.openjudge.cn/practice/05443/ 兔子与樱花

```
inf=float('inf')
p=int(input())
l=[]
dic={}
for i in range(p):
    space=input()
    l.append(space)
    dic[space]=i

q=int(input())
graph=[[inf]*p for i in range(p)]
next=[[-1]*p for i in range(p)]
for i in range(q):
    svs,sve,w=input().split()
    vs,ve=dic[svs],dic[sve]
    if graph[vs][ve]>int(w):
        graph[vs][ve]=graph[ve][vs]=int(w)
        next[vs][ve]=ve
        next[ve][vs]=vs
for i in range(p):
    graph[i][i]=0

for k in range(p):
    for i in range(p):
        for j in range(p):
            dist=graph[i][k] + graph[k][j]
            if graph[i][j]>dist:
                graph[i][j]=dist
                next[i][j]=next[i][k]
def find(i,j):
    if i==j:
        return l[i]
    ans=l[i]
    while next[i][j]!=j:
        sep=next[i][j]
        ans+=f"->({graph[i][sep]})->{l[sep]}"
        i=sep
    ans+=f"->({graph[i][j]})->{l[j]}"
    return ans
r=int(input())
for _ in range(r):
    svs,sve=input().split()
    vs,ve=dic[svs],dic[sve]
    print(find(vs,ve))
```



##### 1.2.2.3 最小生成树

最小生成树的正式定义如下：对于图G=(V, E)，最小生成树T是E的无环子集，并且连接V 中的所有顶点，并且T中边集合的权重之和最小。

###### 1.prim算法

示例：https://leetcode.cn/problems/min-cost-to-connect-all-points/连接所有点的最小费用

```
from heapq import heappush,heapify,heappop
class Solution(object):
    def minCostConnectPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        n=len(points)
        if n<=1:
            return 0
        graph={i:[] for i in range(n)}
        for i in range(n):
            for j in range(i+1,n):
                xi,yi=points[i]
                xj,yj=points[j]
                l=abs(xi-xj)+abs(yi-yj)
                graph[i].append((l,j))
                graph[j].append((l,i))
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
        return ans
```

###### 2.Kruskal算法

```
class disjointset:
    def __init__(self,num):
        self.parent=[i for i in range(num)]
        self.rank=[0]*num
    def find(self,i):
        if self.parent[i]!=i:
            self.parent[i]=self.find(self.parent[i])
        return self.parent[i]
    def union(self,i,j):
        rooti=self.find(i)
        rootj=self.find(j)
        if rooti!=rootj:
            if self.rank[i]<self.rank[j]:
                self.parent[rooti]=rootj
            if self.rank[j]<self.rank[i]:
                self.parent[rootj]=rooti
            else:
                self.parent[rooti]=rootj
                self.rank[j]+=1
n,m=map(int,input().split())
edges=[list(map(int,input().split())) for _ in range(m)]
edges.sort(key=lambda x:x[2])
djset=disjointset(n)
ans=[]
for u,v,w in edges:
    if djset.find(u)!=djset.find(v):
        djset.union(u,v)
        ans.append((u,v,w))
print(*ans)
```

示例：https://leetcode.cn/problems/min-cost-to-connect-all-points/连接所有点的最小费用

```
class disjointset:
    def __init__(self,num):
        self.parent=[i for i in range(num)]
        self.rank=[0]*num
    def find(self,i):
        if self.parent[i]!=i:
            self.parent[i]=self.find(self.parent[i])
        return self.parent[i]
    def union(self,i,j):
        rooti=self.find(i)
        rootj=self.find(j)
        if rooti==rootj:
            return False
        
        if self.rank[i]<self.rank[j]:
            self.parent[rooti]=rootj
        if self.rank[j]<self.rank[i]:
            self.parent[rootj]=rooti
        else:
            self.parent[rooti]=rootj
            self.rank[j]+=1
        return True
class Solution(object):
    def minCostConnectPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        n=len(points)
        if n<=1:
            return 0
        edges=[]
        for i in range(n):
            for j in range(i+1,n):
                xi,yi=points[i]
                xj,yj=points[j]
                l=abs(xi-xj)+abs(yi-yj)
                edges.append([i,j,l])
        edges.sort(key=lambda x:x[2])
        djset=disjointset(n)
        ans=0
        num=0
        for u,v,w in edges:
            if djset.union(u,v): 
                ans+=w
                num+=1
            if num==n:
                break
        return ans
```

### 并查集

[OpenJudge - 28276:判断等价关系是否成立](http://cs101.openjudge.cn/practice/28276/)

```
def find(parent, i):
    if parent[i] == i:
        return i
    parent[i] = find(parent, parent[i])
    return parent[i]


def union(parent, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    parent[rootY] = rootX


m = int(input())
combine = []
varify = []
index = 0
dic = {}

for _ in range(m):
    s = input()
    if not s[0] in dic:
        dic[s[0]] = index
        index += 1
    if not s[-1] in dic:
        dic[s[-1]] = index
        index += 1

    if s[1] == "=":
        combine.append((s[0], s[-1]))

    else:
        varify.append(((s[0], s[-1])))

parent = [i for i in range(index)]
for a, b in combine:
    union(parent, dic[a], dic[b])

flag = True
for a, b in varify:
    if find(parent, dic[a]) == find(parent, dic[b]):
        flag = False
        break

print(flag)
```
