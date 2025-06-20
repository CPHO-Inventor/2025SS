# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

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

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：



代码：

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        

        def ts_to_tree(nums):
            r = len(nums) # l = 0
            m = r // 2
            root = TreeNode(nums[m])
            if r > 2:
                root.left = ts_to_tree(nums[:m])
                root.right = ts_to_tree(nums[m + 1:])
            elif r == 2:
                root.left = ts_to_tree(nums[:1])

            return root
        
        return ts_to_tree(nums)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![6ab4b079ee9d5306dede04e1fb0e1a4](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\6ab4b079ee9d5306dede04e1fb0e1a4.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：



代码：

```python
class TreeNode:
    def __init__(self,val = None):
        self.val = val
        self.children = []

n = int(input())
nodes = {}
flag = {}
for i in range(n):
    s = [int(_) for _ in input().split()]
    for c in s:
        if c not in nodes:
            nodes[c] = TreeNode(c)
            flag[c] = True

    for j in s[1:]:
        nodes[s[0]].children.append(nodes[j])
        flag[j] = False

for k in flag.keys():
    if flag[k] == True:
        root = nodes[k]
        break

def transthrough(root):
    if not root.children:
        return [str(root.val)]

    root.children.sort(key=lambda x: x.val)
    ans = []
    flag = False
    for c in root.children:
        if root.val < c.val and not flag:
            ans.append(str(root.val))
            flag = True
        ans.extend(transthrough(c))

    if not flag:
        ans.append(str(root.val))

    return ans

print("\n".join(transthrough(root)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![03493064f22080d2dbba5a21ce51e1a](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\03493064f22080d2dbba5a21ce51e1a.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：



代码：

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, prevTotal: int) -> int:
            if not root:
                return 0
            total = prevTotal * 10 + root.val
            if not root.left and not root.right:
                return total
            else:
                return dfs(root.left, total) + dfs(root.right, total)

        return dfs(root, 0)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![2f08c3633dd94ae483cb1b1fd57490e](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\2f08c3633dd94ae483cb1b1fd57490e.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：



代码：

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
def build_tree(inorder, postorder):
    if not inorder or not postorder:
        return None
    root_value = postorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(inorder[:root_index_inorder], postorder[1:root_index_inorder + 1])
    root.right = build_tree(inorder[root_index_inorder + 1:], postorder[root_index_inorder + 1:])
    return root
def preorder_traversal(root):
    result = []
    if not root:
        return []
    result += preorder_traversal(root.left)
    result += preorder_traversal(root.right)
    result += [root.value]
    return result

while True:
    try:
        postorder_sequence = input()
        inorder_sequence = input()
        root = build_tree(list(inorder_sequence), list(postorder_sequence))
        preorder_result = preorder_traversal(root)
        print(''.join(preorder_result))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![1f353b1e62e82b639db2f84a7db07c7](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\1f353b1e62e82b639db2f84a7db07c7.png)



### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：



代码：

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点


def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

# 主程序
def main():
    s = input().strip()
    s = ''.join(s.split())
    root = parse_tree(s)  # 解析整棵树
    if root:
        print(preorder(root))  # 输出前序遍历序列
        print(postorder(root))  # 输出后序遍历序列
    else:
        print("input tree string error!")

if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![3a58e8798e7e883912e7273fa04895e](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\3a58e8798e7e883912e7273fa04895e.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：



代码：

```python
import heapq
from typing import List

class Node:
    def __init__(self, val: int, index: int):
        self.val = val
        self.prev = None
        self.next = None
        self.alive = True
        self.index = index

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        # 初始化节点和双向链表
        nodes = [Node(nums[i], i) for i in range(n)]
        for i in range(n):
            if i > 0:
                nodes[i].prev = nodes[i - 1]
            else:
                nodes[i].prev = None
            if i < n - 1:
                nodes[i].next = nodes[i + 1]
            else:
                nodes[i].next = None

        # 计算初始逆序对数
        bad = 0
        for i in range(n - 1):
            if nodes[i].val > nodes[i + 1].val:
                bad += 1

        # 初始化堆
        heap = []
        for i in range(n - 1):
            current_node = nodes[i]
            next_node = current_node.next
            heapq.heappush(heap, (current_node.val + next_node.val, i))

        cnt = 0

        while bad > 0:
            if not heap:
                break  # 堆为空但仍有逆序对，说明逻辑错误

            s, i = heapq.heappop(heap)
            current_node = nodes[i]
            next_node = current_node.next

            # 检查 next_node 是否存在
            if next_node is None:
                continue

            # 跳过无效条目
            if not current_node.alive or not next_node.alive or (current_node.val + next_node.val) != s:
                continue

            prev_node = current_node.prev
            next_next_node = next_node.next

            # 移除旧逆序对
            # 1. prev_node 和 current_node 的逆序
            if prev_node and prev_node.alive and prev_node.val > current_node.val:
                bad -= 1
            # 2. current_node 和 next_node 的逆序
            if current_node.val > next_node.val:
                bad -= 1
            # 3. next_node 和 next_next_node 的逆序
            if next_next_node and next_next_node.alive and next_node.val > next_next_node.val:
                bad -= 1

            # 合并 next_node 到 current_node
            current_node.val += next_node.val
            next_node.alive = False

            # 更新指针
            current_node.next = next_next_node
            if next_next_node:
                next_next_node.prev = current_node
            else:
                current_node.next = None  # 确保指针正确

            # 添加新逆序对
            # 1. prev_node 和 current_node 的新逆序
            if prev_node and prev_node.alive and prev_node.val > current_node.val:
                bad += 1
            # 2. current_node 和 next_next_node 的新逆序
            if next_next_node and next_next_node.alive and current_node.val > next_next_node.val:
                bad += 1

            # 将新邻对推入堆
            if prev_node and prev_node.alive:
                heapq.heappush(heap, (prev_node.val + current_node.val, prev_node.index))
            if next_next_node and next_next_node.alive:
                heapq.heappush(heap, (current_node.val + next_next_node.val, current_node.index))

            cnt += 1

        return cnt
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![4bb89ad44849c553d26f69eedd0966c](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\4bb89ad44849c553d26f69eedd0966c.png)



## 2. 学习总结和收获

期中周刚刚结束，这次作业有的题想不出来就直接deepseek了，下半学期要加大在数算上花的时间。









