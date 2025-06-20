# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>



代码：

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return 2 * (sum(set(nums))) - sum(nums)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![6ee6615717c6b62ea35af84f0bcafc7](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\6ee6615717c6b62ea35af84f0bcafc7.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：



代码：

```python
s = input()
stack = []
for i in range(len(s)):
    stack.append(s[i])
    if stack[-1] == "]":
        stack.pop()
        helpstack = []
        while stack[-1] != "[":
            helpstack.append(stack.pop())
        stack.pop()
        numstr = ""
        while helpstack[-1] in "0123456789":
            numstr += str(helpstack.pop())
        helpstack = helpstack*int(numstr)
        while helpstack != []:
            stack.append(helpstack.pop())
print(''.join(stack))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![a54d192d76f2fdfbbe54610e59f477c](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\a54d192d76f2fdfbbe54610e59f477c.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：



代码：

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        A = headA
        B = headB
        while not A == B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![cee886b6ef9540a414c2d0a3542a903](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\cee886b6ef9540a414c2d0a3542a903.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：



代码：

```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        cur, pre = head, None
        while cur:
            tmp = cur.next 
            cur.next = pre 
            pre = cur      
            cur = tmp      
        return pre
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![8e715659275319089a14875f1b0f17d](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\8e715659275319089a14875f1b0f17d.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：



代码：

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        nums = sorted(zip(nums1, list(range(n))), key=lambda x: x[0])
        heap = []
        pre = []
        acc = 0
        for _, j in nums:
            pre.append(acc)
            heappush(heap, nums2[j])
            acc += nums2[j]
            if len(heap) > k:
                acc -= heappop(heap)
            
        ans = []
        for x in nums1:
            idx = bisect_left(nums, x, key=lambda x: x[0])
            ans.append(pre[idx])
        return ans


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![e0f1cab1998bc89170461e54f871527](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\e0f1cab1998bc89170461e54f871527.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。

![c9d33f2f736db77814f773be2353bd5](C:\Users\18963\Documents\WeChat Files\wxid_7d3wlsw6htiu22\FileStorage\Temp\c9d33f2f736db77814f773be2353bd5.png)



## 2. 学习总结和收获

这周作业就自己做出来两道，备受打击（，感觉神经网络有点意思，有空可以花点时间弄一下。还在补每日选做。









