# from lintcode import (
#     ListNode,
# )
class ListNode:
    def __init__(self,val):
        self.val = val
        self.next = None


# class ListNode:
#     def __init__(self, val):
#         self.val = val ;
#         self.next = None

class MyQueue:

    def __init__(self):
        self.dummy = ListNode(-1)# 初始化哑结点
        self.tail = self.dummy# 头尾相连

    def enqueue(self, item):
        """
        @param: item: An integer
        @return: nothing
        """
        self.tail.next = ListNode(item) #连接新节点
        self.tail = self.tail.next #把tail放到现在的tailnext去，就是让新节点变成了tail

    def dequeue(self):
        """
        @return: An integer
        """
        # write your code here
        #异常检测

        ## 空节点
        if self.dummy.next is None:# 因为dummy是人造的，这里是检测是否为空节点
            return -1
        ## 一个元素
        if self.dummy.next == self.tail:#如果下一个就是tail，那么就一个元素在里面了
            self.tail = self.dummy #tail放到dummy 其实就是变成了
        # 返回值逻辑实现
        ans = self.dummy.next.val #因为是先进先出，dummynext是最开始的第一个元素

        self.dummy.next = self.dummy.next.next #把dummy.next放到现在的next.next，释放第一个元素

        return ans
'''~~~~~~~~~~~~~~~~~~~~~~'''

"""
Definition of ListNode:
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
'''
可划分为三步：

1.找到原链表的中点。
    [我们可以使用快慢指针来 O(N) 地找到链表的中间节点。]
2.将原链表的右半端反转。
    [我们可以使用迭代法实现链表的反转。]
3.将原链表的两端合并。
    [因为两链表长度相差不超过 11，因此直接合并即可]
'''

class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return
        #寻找中点

        mid = self.middleNode(head)
        l1 = head #前半链表
        l2 = mid.next #重点开始
        mid.next = None #因为要反转链表，所以mid会变成最后一个节点，其next应为空
        # 翻转后半段链表
        l2 = self.reverseList(l2)
        # 合并链表
        self.mergeList(l1, l2)

    def middleNode(self, head: ListNode) -> ListNode:
        #fast是slow的2倍速度，最后slow刚好在中点
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverseList(self, head: ListNode) -> ListNode:
        '''
        翻转后半部分链表

        把nextTemp放到现在的curr.next上面，存起来避免丢失信息
        -》因为要逆转，所以把curr.next放到现在的prev
        -》这个节点逆转完了，要逆转下一个节点了，那么本节点相对于下个节点就是prev了，所以把pre放到现在的节点cur上
        -》现在的cur要放到下一个节点，也就是存起起来的nextTemp上
        '''
        prev = None
        curr = head
        while curr:
            nextTemp = curr.next # nextTemp放现在的curr.next
            curr.next = prev # curr.next放现在的prev
            prev = curr # prev放现在的curr
            curr = nextTemp # curr放现在的nextTemp
        return prev

    def mergeList(self, l1: ListNode, l2: ListNode):
        '''
        这里的逻辑和上面翻转链表的差不多，只不过是交叉性质的
        '''
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l1 = l1_tmp

            l2.next = l1
            l2 = l2_tmp

'''------------------------------------------'''
def rotateRight(self, head, k):
    # write your code here
    #异常检测
    if not head or not head.next:
        return head

    # calculate the len
    len = 0 #计数长度
    tail = head
    # 走到尾部 并计数
    while tail: #
        tail = tail.next
        len += 1

    #k取值对节点位置影响的处理
    k = k % len # k > len时实则k-len才是真正的位置
    if k == 0:
        return head

    #制作快慢指针
    slow, fast = head, head
    for _ in range(k):#因为结果是要距离尾部的k，所以fast比slow快k
        fast = fast.next

    while fast.next:
        slow = slow.next
        fast = fast.next

    new_head = slow.next
    slow.next = None
    fast.next = head


    dummy = ListNode(0, slow.next)
    slow.next = None
    fast.next = head

    return dummy.next


    return new_head


'''~~~~~~~~~~~~~~~~'''

class Solution:
    """
    @param head: the head of linked list.
    @return: a middle node of the linked list
    思路：
        用快慢指针
    存在问题：
        如何构造这个快慢指针，p.next.next的话，出现超界时，显示NONETYPE has no attribute “next”
    解决方法：
        slow = head，fast = slow.next  然后while fast and fast.next:保证到倒数第二个节点停止
    """

    class Solution:
        def middle_node(self, head: ListNode) -> ListNode:
            if not head:
                return head
            A = [head]
            while A[-1].next:
                A.append(A[-1].next)
            return A[(len(A) - 1) // 2]

            A = []
            dummy = ListNode(0, head)
            while dummy.next:
                print(dummy.next.val)
                A.append(dummy.next)
            return A[(len(A) - 1) // 2]

    class Solution:
        # @param head: the head of linked list.
        # @return: a middle node of the linked list
        def middleNode(self, head):
            # Write your code here
            if head is None:
                return None
            slow = head
            fast = slow.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next

            return slow


'''---------------------------'''

"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""


class Solution:
    """
    @param head: a ListNode
    @param val: An integer
    @return: a ListNode
    """

    def removeElements(self, head, val):
        # write your code here
        dummy = ListNode(-1, head)
        p = dummy
        while p.next:
            if p.next.val == val:
                p.next = p.next.next
            else:
                p = p.next
        return dummy.next


'''---------------------------'''

'''交换节点
1 找到对应两个节点值，进行交换  
2 节点的值都不相同
2 如果找不到任一节点，那么什么不用做
4 交换节点而非改值

思路：
# 异常检测（空链表）
# 寻找逻辑
## 指针移动-》以val建立字典-》 next.val in val_dict？ ->根据key放入p.next和p.next.next + 对找到元素个数计数
-》两个都找到时 进行交换

存在问题：
    理论上用的赋值形式，修改时问最后交换的是内存地址，没有对 

'''


class Solution:
    """
    @param head: a ListNode
    @param v1: An integer
    @param v2: An integer
    @return: a new head of singly-linked list
    """

    def swap_nodes(self, head: ListNode, v1: int, v2: int) -> ListNode:
        dummy = ListNode(0, head)
        # 异常检测
        if dummy.next == None:
            return dummy.next
        p = dummy
        count = 0
        val_dict = {v1: ListNode(0), v2: ListNode(0)}

        ## 寻找符合条件节点
        while p.next != None:
            value = p.next.val
            if value in val_dict.keys():
                val_dict[value] = p
                count += 1
            if count == 2:
                # 交换内容
                val_dict[v1].next, val_dict[v2].next = val_dict[v2].next, val_dict[v1].next
                val_dict[v1].next.next, val_dict[v2].next.next = val_dict[v2].next.next, val_dict[v1].next.next
                return dummy.next
            p = p.next
        return dummy.next


'''
提供一个不用区分v1, v2所在Node是否相连的做法：
找到preV1, preV2后，先换入口再换出口：
把preV1.next, preV2.next互换，preV1.next.next, preV2.next.next互换 
———— 此处顺序不可相反，否则v1, v2相连的case会出问题
'''


class Solution:
    """
    @param head: a ListNode
    @param v1: An integer
    @param v2: An integer
    @return: a new head of singly-linked list
    """

    def swapNodes(self, head, v1, v2):
        # find preV1 and preV2
        # swap (preV1.next, preV2.next) then swap (preV1.next.next, preV2.next.next)
        dummy = ListNode(0, head)
        preV1 = dummy
        preV2 = dummy
        # 找到v1节点
        while preV1.next and preV1.next.val != v1:  # 链表未到尽头，且下一节点的值不等于v1
            preV1 = preV1.next
        # 如果没找到返回head
        if not preV1.next:
            return dummy.next
        while preV2.next and preV2.next.val != v2:
            preV2 = preV2.next
        if not preV2.next:
            return dummy.next
        preV1.next, preV2.next = preV2.next, preV1.next
        preV1.next.next, preV2.next.next = preV2.next.next, preV1.next.next
        return dummy.next

        # write your code here


# .header-1yNTfz5xx_WeoHZWN5Am4c


if __name__ == '__main__':
    print(1)
