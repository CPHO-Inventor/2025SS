# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

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

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：



代码：

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = dum = ListNode(0)
        while list1 and list2:
            if list1.val < list2.val:
                cur.next, list1 = list1, list1.next
            else:
                cur.next, list2 = list2, list2.next
            cur = cur.next
        cur.next = list1 if list1 else list2
        return dum.next
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![565be83bb8b0e04add61de29e889841](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\565be83bb8b0e04add61de29e889841.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>



代码：

```python
class Solution:

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    # 206. 反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        mid = self.middleNode(head)
        head2 = self.reverseList(mid)
        while head2:
            if head.val != head2.val:  # 不是回文链表
                return False
            head = head.next
            head2 = head2.next
        return True
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![44a82cf911f0bdd16ecc926a374d278](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\44a82cf911f0bdd16ecc926a374d278.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双向链表实现。</mark>



代码：

```python
class BrowserHistory:

    def __init__(self, homepage: str):
        self.stk1 = []
        self.stk2 = []
        self.visit(homepage)


    def visit(self, url: str) -> None:
        self.stk2.clear()
        self.stk1.append(url)

    def back(self, steps: int) -> str:
        while steps and len(self.stk1) > 1:
            self.stk2.append(self.stk1.pop())
            steps -= 1
        return self.stk1[-1]

    def forward(self, steps: int) -> str:
        while steps and self.stk2:
            self.stk1.append(self.stk2.pop())
            steps -= 1
        return self.stk1[-1]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![a76b45a66b1aaeef133d998d40d1830](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a76b45a66b1aaeef133d998d40d1830.png)



### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：



代码：

```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''

    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()

    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)

    while stack:
        postfix.append(stack.pop())

    return ' '.join(str(x) for x in postfix)

n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![82db1cb026b797faa53c0d3df2e68e7](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\82db1cb026b797faa53c0d3df2e68e7.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>



代码：

```python
while True:
    list0 = input().split()
    n = int(list0[0])
    p = int(list0[1])
    m = int(list0[2])
    if n == 0 and p == 0 and m == 0:
        break
    else:
        list1 = [i for i in range(1, n + 1)]
        num = p - 1
        yes_or_no = True
        list2 = []
        while yes_or_no:
            noun = list1.pop()
            if list1:
                list1.append(noun)
                d = (m - 1) % len(list1)
                y = num + d
                if y < len(list1) - 1:
                    list2.append(str(list1[y]))
                    del list1[y]
                    num = y
                elif y > len(list1) - 1:
                    z = y - (len(list1) - 1) - 1
                    list2.append(str(list1[z]))
                    del list1[z]
                    num = z
                else:
                    list2.append(str(list1[y]))
                    del list1[y]
                    num = 0
            else:
                list2.append(str(noun))
                yes_or_no = False
        print(','.join(list2))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![a7f805b71af7de4366246778625c14d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a7f805b71af7de4366246778625c14d.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：



代码：

```python
def merge_and_count_split_inversions(left, right):

    merged = []
    inversions = 0
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inversions += len(left) - i
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged, inversions


def sort_and_count_inversions(sequence):

    if len(sequence) <= 1:
        return sequence, 0

    mid = len(sequence) // 2
    left, left_inversions = sort_and_count_inversions(sequence[:mid])
    right, right_inversions = sort_and_count_inversions(sequence[mid:])
    merged, split_inversions = merge_and_count_split_inversions(left, right)

    total_inversions = left_inversions + right_inversions + split_inversions
    return merged, total_inversions


while True:
    try:
        n = int(input())
        list1 = []
        for i in range(n):
            list1.append(int(input()))
        list0 = [0] * n
        for i in range(n):
            list0[i] = list1[~i]
        a, b = sort_and_count_inversions(list0)
        input()
        print(b)
        print('')
    except:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![715df5f24429f6cf6c2a61bbd2f12b4](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\715df5f24429f6cf6c2a61bbd2f12b4.png)



## 2. 学习总结和收获

做完这次作业感觉还是对链表不太熟悉，但对排序和堆更熟悉了一些，在补每日选做











