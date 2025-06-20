# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

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

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：



代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(x):
            if x == len(nums) - 1:
                res.append(list(nums))   
                return
            for i in range(x, len(nums)):
                nums[i], nums[x] = nums[x], nums[i]  
                dfs(x + 1)                          
                nums[i], nums[x] = nums[x], nums[i] 
        res = []
        dfs(0)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![bf7c5e1321f0d7c5743cd92d0ab4854](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\bf7c5e1321f0d7c5743cd92d0ab4854.png)

### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：



代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        def dfs(i: int, j: int, k: int) -> bool:
            if board[i][j] != word[k]:  # 匹配失败
                return False
            if k == len(word) - 1:  # 匹配成功！
                return True
            board[i][j] = ''  # 标记访问过
            for x, y in (i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j):  # 相邻格子
                if 0 <= x < m and 0 <= y < n and dfs(x, y, k + 1):
                    return True  # 搜到了！
            board[i][j] = word[k]  # 恢复现场
            return False  # 没搜到
        return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![e531f5b949db790e55b0a51f1a38852](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\e531f5b949db790e55b0a51f1a38852.png)

### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：



代码：

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401125807296](C:\Users\18963\AppData\Roaming\Typora\typora-user-images\image-20250401125807296.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：



代码：

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        ans = []
        cur = [root]
        while cur:
            nxt = []
            vals = []
            for node in cur:
                vals.append(node.val)
                if node.left:  nxt.append(node.left)
                if node.right: nxt.append(node.right)
            cur = nxt
            ans.append(vals)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![90ae9701219f7132c5bc0f4c86b362d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\90ae9701219f7132c5bc0f4c86b362d.png)

### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：



代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
  
        n = len(s)
        ans = []
        path = []

      
        def dfs(i: int) -> None:
            if i == n:  
                ans.append(path.copy()) 
                return
            for j in range(i, n):  
                t = s[i: j + 1] 
                if t == t[::-1]: 
                    path.append(t)
                   
                    dfs(j + 1)
                    path.pop() 

        dfs(0)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![b3271c16b2525037ec9dc952d9b512d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\b3271c16b2525037ec9dc952d9b512d.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：



代码：

```python
class Node:
        def __init__(self, key: int, val: int):
            self.key = key 
            self.val = val
            self.prev = None 
            self.next = None 

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}
        self.cnt = 0
        self.head = Node(-1, -1)
        self.tail = Node(-1, -1)
        self.head.next = self.tail 
        self.tail.prev = self.head 
    
    def remove(self, node: Node) -> None:
        node.prev.next = node.next 
        node.next.prev = node.prev 

    def add(self, node: Node) -> None:
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node 
        node.prev = self.head 

    def get(self, key: int) -> int:
        if key in self.map:
            node = self.map[key]
            self.remove(node)
            self.add(node)
            return node.val
        return -1 

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            node = self.map[key]
            node.val = value 
            self.remove(node)
            self.add(node)
            self.map[key] = node
        else:
            node = Node(key, value)
            self.add(node)
            self.map[key] = node 
            if self.cnt == self.capacity:
                del_node = self.tail.prev
                self.remove(del_node)
                del self.map[del_node.key]
            else:
                self.cnt += 1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![46ae5a9d15ce84a92e25f468eab957b](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\46ae5a9d15ce84a92e25f468eab957b.png)



## 2. 学习总结和收获

对树的层序中序有了一定的理解，回顾了一下bfs,dfs。马上其中考试了，每日选做可能要暂停一下。











