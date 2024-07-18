import asyncio
import json
import os
import uuid

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, trace
from parea.schemas import Completion, LLMInputs, Message, Role, Log, EvaluationResult

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name="LeetCode")
p.wrap_openai_client(client)

MODELS = ["codestral-latest", "codestral-mamba-latest"]

merge_nodes_in_between_zeros_2181 = """You are given the head of a linked list, which contains a series of integers separated by 0's. The beginning and end of the linked list will have Node.val == 0.

For every two consecutive 0's, merge all the nodes lying in between them into a single node whose value is the sum of all the merged nodes. The modified list should not contain any 0's.

Return the head of the modified linked list.


Example 1:


Input: head = [0,3,1,0,4,5,2,0]
Output: [4,11]
Explanation:
The above figure represents the given linked list. The modified list contains
- The sum of the nodes marked in green: 3 + 1 = 4.
- The sum of the nodes marked in red: 4 + 5 + 2 = 11.
Example 2:


Input: head = [0,1,0,3,0,2,2,0]
Output: [1,3,4]
Explanation:
The above figure represents the given linked list. The modified list contains
- The sum of the nodes marked in green: 1 = 1.
- The sum of the nodes marked in red: 3 = 3.
- The sum of the nodes marked in yellow: 2 + 2 = 4.


Constraints:

The number of nodes in the list is in the range [3, 2 * 105].
0 <= Node.val <= 1000
There are no two consecutive nodes with Node.val == 0.
The beginning and end of the linked list have Node.val == 0."""
number_of_good_leaf_nodes_pairs_1530 = """
You are given the root of a binary tree and an integer distance. A pair of two different leaf nodes of a binary tree is said to be good if the length of the shortest path between them is less than or equal to distance.

Return the number of good leaf node pairs in the tree.



Example 1:


Input: root = [1,2,3,null,4], distance = 3
Output: 1
Explanation: The leaf nodes of the tree are 3 and 4 and the length of the shortest path between them is 3. This is the only good pair.
Example 2:


Input: root = [1,2,3,4,5,6,7], distance = 3
Output: 2
Explanation: The good pairs are [4,5] and [6,7] with shortest path = 2. The pair [4,6] is not good because the length of ther shortest path between them is 4.
Example 3:

Input: root = [7,1,4,6,null,5,3,null,null,null,null,null,2], distance = 3
Output: 1
Explanation: The only good pair is [2,5].


Constraints:

The number of nodes in the tree is in the range [1, 210].
1 <= Node.val <= 100
1 <= distance <= 10
"""
partitioning_into_minimum_number_of_deci_binary_numbers_1689 = """
A decimal number is called deci-binary if each of its digits is either 0 or 1 without any leading zeros. For example, 101 and 1100 are deci-binary, while 112 and 3001 are not.

Given a string n that represents a positive decimal integer, return the minimum number of positive deci-binary numbers needed so that they sum up to n.



Example 1:

Input: n = "32"
Output: 3
Explanation: 10 + 11 + 11 = 32
Example 2:

Input: n = "82734"
Output: 8
Example 3:

Input: n = "27346209830709182346"
Output: 9


Constraints:

1 <= n.length <= 105
n consists of only digits.
n does not contain any leading zeros and represents a positive integer.
"""
insert_greatest_common_divisors_in_linked_list_2807 = """
Given the head of a linked list head, in which each node contains an integer value.

Between every pair of adjacent nodes, insert a new node with a value equal to the greatest common divisor of them.

Return the linked list after insertion.

The greatest common divisor of two numbers is the largest positive integer that evenly divides both numbers.



Example 1:


Input: head = [18,6,10,3]
Output: [18,6,6,2,10,1,3]
Explanation: The 1st diagram denotes the initial linked list and the 2nd diagram denotes the linked list after inserting the new nodes (nodes in blue are the inserted nodes).
- We insert the greatest common divisor of 18 and 6 = 6 between the 1st and the 2nd nodes.
- We insert the greatest common divisor of 6 and 10 = 2 between the 2nd and the 3rd nodes.
- We insert the greatest common divisor of 10 and 3 = 1 between the 3rd and the 4th nodes.
There are no more adjacent nodes, so we return the linked list.
Example 2:


Input: head = [7]
Output: [7]
Explanation: The 1st diagram denotes the initial linked list and the 2nd diagram denotes the linked list after inserting the new nodes.
There are no pairs of adjacent nodes, so we return the initial linked list.


Constraints:

The number of nodes in the list is in the range [1, 5000].
1 <= Node.val <= 1000
"""
binary_search_tree_to_greater_sum_tree_1038 = """
Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus the sum of all keys greater than the original key in BST.

As a reminder, a binary search tree is a tree that satisfies these constraints:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.


Example 1:


Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
Example 2:

Input: root = [0,null,1]
Output: [1,null,1]


Constraints:

The number of nodes in the tree is in the range [1, 100].
0 <= Node.val <= 100
All the values in the tree are unique.
"""

answer_2181 = """
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr=head.next, head.next
        sum=0
        while curr:
            x=curr.val
            if x!=0: sum+=x
            else:
                prev.val=sum
                prev.next=curr.next
                prev=prev.next
                sum=0
            curr=curr.next
        return head.next
        """
answer_1530 = """
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        self.map = {}
        self.leaves = []
        self.findLeaves(root, [], self.leaves, self.map)
        res = 0
        for i in range(len(self.leaves)):
            for j in range(i + 1, len(self.leaves)):
                list_i, list_j = self.map[self.leaves[i]], self.map[self.leaves[j]]
                for k in range(min(len(list_i), len(list_j))):
                    if list_i[k] != list_j[k]:
                        dist = len(list_i) - k + len(list_j) - k
                        if dist <= distance:
                            res += 1
                        break
        return res

    def findLeaves(self, node: TreeNode, trail: List[TreeNode], leaves: List[TreeNode], map: Dict[TreeNode, List[TreeNode]]):
        if not node:
            return
        tmp = trail[:]
        tmp.append(node)
        if not node.left and not node.right:
            map[node] = tmp
            leaves.append(node)
            return
        self.findLeaves(node.left, tmp, leaves, map)
        self.findLeaves(node.right, tmp, leaves, map)"""
answer_1689 = """
class Solution:
    def minPartitions(self, n: str) -> int:
        m=0
        for i in n:
            if int(i)>m:
                m=int(i)
        return m;"""
answer_2807 = """
class Solution:
    def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while(cur.next):
            cur.next = ListNode(gcd(cur.val,cur.next.val), cur.next)
            cur = cur.next.next
        return head"""
answer_1038 = """
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        ans=0
        def helper(node):
            nonlocal ans
            if node==None:
                return
            helper(node.right)
            ans+=node.val
            node.val=ans
            helper(node.left)
        helper(root)
        return root
"""

leet_code_questions = [
    merge_nodes_in_between_zeros_2181,
    number_of_good_leaf_nodes_pairs_1530,
    partitioning_into_minimum_number_of_deci_binary_numbers_1689,
    insert_greatest_common_divisors_in_linked_list_2807,
    binary_search_tree_to_greater_sum_tree_1038,
]
leet_code_answers = [answer_2181, answer_1530, answer_1689, answer_2807, answer_1038]

data = [{"question": q, "target": t} for q, t in zip(leet_code_questions, leet_code_answers)]


def correct_code(log: Log) -> EvaluationResult:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"""[Instruction]\nPlease act as an impartial code grader and evaluate the quality and
                    correctness of the response provided. Your evaluation should consider factors such as the
                    correctness, code quality, performance, depth, creativity, and level of detail of the response.
                    Be as objective as possible. Respond in JSON with two fields: \n
                    \t 1. score: int = a number from a scale of 0 to 5 with 5 being great and 0 being bad.\n
                    \t 2. reason: str =  explain your reasoning for the selected score.\n\n
                    This is this question asked: QUESTION:\n{log.inputs["question"]}\n
                    This is an example solution: Example Solution:\n{log.target}\n
                    This is the response you are grading, RESPONSE:\n{log.output}\n\n""",
                }
            ],
            response_format={"type": "json_object"},
        )
        r = json.loads(response.choices[0].message.content)
        return EvaluationResult(name="correct_code", score=int(r["score"]) / 5, reason=r["reason"])
    except Exception as e:
        return EvaluationResult(name="error-correct_code", score=0, reason=f"Error in grading: {e}")


def model_call_factory(model: str):
    @trace(eval_funcs=[correct_code])
    def answer_lc(question: str) -> str:
        return p.completion(
            data=Completion(
                llm_configuration=LLMInputs(
                    model=model,
                    messages=[Message(role=Role.user, content=f"Answer this coding interview question in Python: {question}\n")],
                ),
                cache=False,
            )
        ).content

    return answer_lc


async def main():
    await asyncio.gather(*[p.experiment(name="LeetCodeQuestions", data=data, func=model_call_factory(model)).arun(run_name=f"{model}-{str(uuid.uuid4())[:4]}") for model in MODELS])


if __name__ == "__main__":
    asyncio.run(main())
