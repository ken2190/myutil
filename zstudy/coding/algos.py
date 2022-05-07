# -*- coding: utf-8 -*-



##############################################################################################
##############################################################################################
Given a string S, find the longest palindromic substring in S. 
Substring of string S: S[ i . . . . j ] where 0 ≤ i ≤ j < len(S). 
Palindrome string: A string which reads the same backwards. 
More formally, S is palindrome if reverse(S) = S. Incase of conflict, 
return the substring which occurs first ( with the least starting index).


Example 1:

Input:
S = "aaaabbaa"
Output: aabbaa
Explanation: The longest Palindromic
substring is "aabbaa".
Example 2:

Input: 
S = "abc"
Output: a
Explanation: "a", "b" and "c" are the 
longest palindromes with same length.
The result is the one with the least
starting index.


# A O(n ^ 2) time and O(1) space program to find the# longest palindromic substring

def longestPalSubstr(string):
    n = len(string) # calculating size of string
    if (n < 2):
        return n # if string is empty then size will be 0. # if n==1 then, answer will be 1(single



    start=0
    maxLength = 1 
    for i in range(n):
        low  = i - 1
        high = i + 1

        
        while (high < n and string[high] == string[i] ):                               
            high=high+1
      
        while (low >= 0 and string[low] == string[i] ):                 
            low=low-1
      
        while (low >= 0 and high < n and string[low] == string[high] ):
          low=low-1
          high=high+1 
        
    
        length = high - low - 1
        if (maxLength < length):
            maxLength = length
            start=low+1
            
    print ("Longest palindrome substring is:",end=" ")
    print (string[start:start + maxLength])
    
    return maxLength
    
# Driver program to test above functions
string = ("forgeeksskeegfor")
print("Length is: " + str(longestPalSubstr(string)))


















##############################################################################################
##############################################################################################
###  1
You are given a string s. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, the resultant string should be s.

Return a list of integers representing the size of these parts.

 

Example 1:

Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits s into less parts.
Example 2:

Input: s = "eccbbbbdec"
Output: [10]
 

Constraints:

1 <= s.length <= 500
s consists of lowercase English letters.



from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:

        def isused(x0, ll):
            ii = -1
            for wi in ll :
                ii = ii + 1
                if x0 in wi : return ii
            return -1
        
        
        res   = ['']
        wordk = 0
        for i,x in enumerate(s) :
            print(i, wordk,x, res)
            
            ix = isused(x, res)
            if len(res[wordk]) ==0:
                res[wordk] = res[wordk] + x
        
            
            elif  ix > -1  :
                if wordk > 0 : ### Merge previous
                   # break 
                   res[ix] = "".join( res[ix:])  + x
                   res   = res[:ix+1]
                   wordk = ix
        
                else :
                    res[wordk] = res[wordk]+ x
                
            else  :
                wordk= wordk +1
                res.append(x)
        
        rcount = [ len(word) for word in res ]    

        return rcount 
            


s = "ababcbacadefegdehijhklij"
    
slist =[ "ababcbacadefegdehijhklij"
        
        ]

for s in slist:
  res = Solution().partitionLabels(s)  
  print(s, ":", res)

    
##########  Debug
def isused(x0, ll):
    ii = -1
    for wi in ll :
        ii = ii + 1
        if x0 in wi : return ii
    return -1


res   = ['']
wordk = 0
for i,x in enumerate(s) :
    print(i, wordk,x, res)
    
    ix = isused(x, res)
    if len(res[wordk]) ==0:
        res[wordk] = res[wordk] + x

    
    elif  ix > -1  :
        if wordk > 0 : ### Merge previous
           # break 
           res[ix] = "".join( res[ix:])  + x
           res   = res[:ix+1]
           wordk = ix

        else :
            res[wordk] = res[wordk]+ x
        
    else  :
        wordk= wordk +1
        res.append(x)
        
    
rcount = [ len(word) for word in res ]    

    
    
    
    
    
    
    
    
    
    
    
    
    

###############################################################  
#################  2
An axis-aligned rectangle is represented as a list [x1, y1, x2, y2], where (x1, y1) 
is the coordinate of its bottom-left corner, and (x2, y2) is the coordinate of its top-right corner. Its top and bottom edges are parallel to the X-axis, and its left and right edges are parallel to the Y-axis.

Two rectangles overlap if the area of their intersection is positive. To be clear, two rectangles that only touch at the corner or edges do not overlap.

Given two axis-aligned rectangles rec1 and rec2, return true if they overlap, otherwise return false.

 

Example 1:

Input: rec1 = [0,0,2,2], rec2 = [1,1,3,3]
Output: true
Example 2:

Input: rec1 = [0,0,1,1], rec2 = [1,0,2,1]
Output: false
Example 3:

Input: rec1 = [0,0,1,1], rec2 = [2,2,3,3]
Output: false
 

Constraints:

rect1.length == 4
rect2.length == 4
-109 <= rec1[i], rec2[i] <= 109
rec1 and rec2 represent a valid rectangle with a non-zero area.


from typing import List
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
          """
             One point should inside of other rectangle.
             Use Symmetry
          
          """

          x1a, y1a, x2a,y2a  = rec1[0], rec1[1], rec1[2], rec1[3]
          x1b, y1b, x2b,y2b  = rec2[0], rec2[1], rec2[2], rec2[3]
            
            
          if   (y1b < y1a < y2b) and  (x1b < x1a < x2b ): return True
        
          elif (y1b < y1a < y2b) and  (x1b < x2a < x2b ): return True        
        

          elif (y1b < y2a < y2b) and  (x1b < x1a < x2b ): return True
            
            
          elif (y1b < y2a < y2b) and  (x1b < x2a < x2b ): return True
            
          
          ### reverse         
          x1a, y1a, x2a,y2a  = rec2[0], rec2[1], rec2[2], rec2[3]
          x1b, y1b, x2b,y2b  = rec1[0], rec1[1], rec1[2], rec1[3]
            
            
          if   (y1b < y1a < y2b) and  (x1b < x1a < x2b ): return True
        
          elif (y1b < y1a < y2b) and  (x1b < x2a < x2b ): return True        
        
          elif (y1b < y2a < y2b) and  (x1b < x1a < x2b ): return True
                        
          elif (y1b < y2a < y2b) and  (x1b < x2a < x2b ): return True
            
          return False  


### Symmetric cases    

rec1 = [0,0,2,2]; rec2 = [1,1,3,3]
#Output: true
Solution().isRectangleOverlap(rec1, rec2 ) 

rec2 = [0,0,2,2]; rec1 = [1,1,3,3]
Solution().isRectangleOverlap(rec1, rec2 ) 



rec1 = [0,0,1,1]; rec2 = [1,0,2,1]
#Output: false
Solution().isRectangleOverlap(rec1, rec2 ) 


rec2 = [0,0,1,1]; rec1 = [1,0,2,1]
Solution().isRectangleOverlap(rec1, rec2 ) 




rec1 = [0,0,1,1]; rec2 = [2,2,3,3]
# Output: false
Solution().isRectangleOverlap(rec1, rec2 ) 
    
    
    
    
    
rec1 = [0,0,10,10]; rec2 = [2,2,10,11]
Solution().isRectangleOverlap(rec1, rec2 ) 
    
    
    
    
    
    
    
    
A string s can be partitioned into groups of size k using the following procedure:

The first group consists of the first k characters of the string, the second group consists of the next k characters of the string, and so on. Each character can be a part of exactly one group.
For the last group, if the string does not have k characters remaining, a character fill is used to complete the group.
Note that the partition is done so that after removing the fill character from the last group (if it exists) and concatenating all the groups in order, the resultant string should be s.

Given the string s, the size of each group k and the character fill, return a string array denoting the composition of every group s has been divided into, using the above procedure.

 

Example 1:

Input: s = "abcdefghi", k = 3, fill = "x"
Output: ["abc","def","ghi"]
Explanation:
The first 3 characters "abc" form the first group.
The next 3 characters "def" form the second group.
The last 3 characters "ghi" form the third group.
Since all groups can be completely filled by characters from the string, we do not need to use fill.
Thus, the groups formed are "abc", "def", and "ghi".
Example 2:

Input: s = "abcdefghij", k = 3, fill = "x"
Output: ["abc","def","ghi","jxx"]
Explanation:
Similar to the previous example, we are forming the first three groups "abc", "def", and "ghi".
For the last group, we can only use the character 'j' from the string. To complete this group, we add 'x' twice.
Thus, the 4 groups formed are "abc", "def", "ghi", and "jxx".


class Solution:
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        res = []
        wordk=-1 ;  ii = -1
        for x in s :
            ii = ii + 1 
            print(ii, x, wordk, res )
            
            if len(res) == 0 :
                res.append(x)
                wordk = wordk+1
                
            elif len(res[wordk])< k :
                res[wordk] +=  x
                
            else :
                res.append(x)
                wordk = wordk+ 1
         
        if len(res[-1])< k :
            res[-1] = res[-1] + fill*(k-len(res[-1]))
        
        return res
        

    
    
s = "abcdefg"; k = 3; fill = "x"


Solution().divideString( s, k, fill)

    



    
    
    
##### 2104. Sum of Subarray Ranges
You are given an integer array nums.
 The range of a subarray of nums is the difference between the largest and smallest element in the subarray.

Return the sum of all subarray ranges of nums.
A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,2,3]
Output: 4
Explanation: The 6 subarrays of nums are the following:
[1], range = largest - smallest = 1 - 1 = 0 
[2], range = 2 - 2 = 0
[3], range = 3 - 3 = 0
[1,2], range = 2 - 1 = 1
[2,3], range = 3 - 2 = 1
[1,2,3], range = 3 - 1 = 2
So the sum of all ranges is 0 + 0 + 0 + 1 + 1 + 2 = 4.

Example 2:
Input: nums = [1,3,3]
Output: 4
Explanation: The 6 subarrays of nums are the following:
[1], range = largest - smallest = 1 - 1 = 0
[3], range = 3 - 3 = 0
[3], range = 3 - 3 = 0
[1,3], range = 3 - 1 = 2
[3,3], range = 3 - 3 = 0
[1,3,3], range = 3 - 1 = 2
So the sum of all ranges is 0 + 0 + 0 + 2 + 0 + 2 = 4.
Example 3:

Input: nums = [4,-2,-3,4,1]
Output: 59
Explanation: The sum of all subarray ranges of nums is 59.
 

Constraints:

1 <= nums.length <= 1000
-109 <= nums[i] <= 109
    
    
#### Sub optimal    

class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:


        def get_subarray(ll, kmax=0):
            res = []
            for i in range(0, len(ll)-kmax):
                lli = ll[i:(i+kmax)]
                if len(lli) == kmax:
                   res.append(lli)
            return res    
        
        n= len(nums)
        ss= 0 
        for sizek in range(2, n+1):
          lallk = get_subarray(nums, kmax=sizek)
          print(lallk)
          for lk in lallk:
             ss = ss + max(lk) - min(lk)
             
        return ss

         

        
### list of all sub-array:

from itertools import permutations, combinations


nums = [1,2,3]


nums = [4,-2,-3,4,1]


nums = [1,3,3]

def get_subarray(ll, kmax=0):
    res = []
    for i in range(0, len(ll)):
        lli = ll[i:(i+kmax)]
        if len(lli) == kmax:
           res.append(lli)
    return res    



n= len(nums)
ss= 0 
for sizek in range(2, n+1):
  lallk = get_subarray(nums, kmax=sizek)
  print(lallk)
  for lk in lallk:
     ss = ss + max(lk) - min(lk)
        
print(ss)







#### 844. Backspace String Compare
Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.
Note that after backspacing an empty text, the text will continue empty.

 

Example 1:

Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".
Example 2:

Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both s and t become "".
Example 3:

Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c" while t becomes "b".
 

Constraints:

1 <= s.length, t.length <= 200
s and t only contain lowercase letters and '#' characters.



















##############################
Permutation is an arrangement of objects in a specific order. 

def permutation(lst):
    # Python function to print permutations of a given list

    # If lst is empty then there are no permutations
    if len(lst) == 0:
        return []
 
    # If there is only one element in lst then, only
    # one permutation is possible
    if len(lst) == 1:
        return [lst]

 
    l = [] 
    for i in range(len(lst)):
       m = lst[i]
 
       # Extract lst[i] or m from the list.  remLst is remaining list
       remLst = lst[:i] + lst[i+1:]
 
       # Generating all permutations where m is first
       # element
       for p in permutation(remLst):
           l.append([m] + p)
    return l
 
 
# Driver program to test above function
data = list('123')
for p in permutation(data):
    print (p)
    
    
    
    
########################
    
You are also given three integers sr, sc, and newColor. 
You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, 
plus any pixels connected 4-directionally to the starting pixel of the same color 
as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with newColor.

Return the modified image after performing the flood fill.



Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are colored with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.



Input: image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, newColor = 2
Output: [[2,2,2],[2,2,2]]



class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        n = len(image)
        import copy
        img2 = copy.deepcopy(image)


        x0,y0 = sr,sc
        c0 = newColor


        xmax, ymax   = n-1,n-1
        xmin, ymin   = 0,0
        img2[x0][y0] = newColor


        def fill_1step(img, x0, y0):

            if img[x0][y0] == 0 : return img

            img[min(xmax,x0+1)][y0] == c0
            img[max(xmin,x0-1)][y0] == c0
            img[x0][min(ymax,y0+1)] == c0
            img[y0][max(ymin,y0-1)] == c0
            return img
        
        def get_new(x0,y0):
            s =[ (min(xmax,x0+1), y0),   (max(xmin,x0-1),y0),
              (x0,  min(ymax,y0+1)),  (x0, max(ymin,y0-1) )
            ]
            return s

        print(img2) 
        
        for  i in range(-n//2, n//2):
           for j in range(-n//2, n//2):
               xk,yk = x0+i, y0+j
               for xi,yi in get_new(xk,yk):
                  img2 = fill_1step(img2, xi, yi)
                
                
        return img2

        



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sufficientSubset(self, root: Optional[TreeNode], limit: int) -> Optional[TreeNode]:
        

Input: root = [1,2,3,4,-99,-99,7,8,9,-99,-99,12,13,-99,14], limit = 1
Output: [1,2,3,4,null,null,7,8,9,null,14]





Input: root = [5,4,8,11,null,17,4,7,1,null,null,5,3], limit = 22
Output: [5,4,8,11,null,17,4,7,null,null,null,5]








### Palindrome
def is_palindrome(s: str) -> bool:
    # Lowercase the string 
    s = s.lower()
    
    # All lowercase letters and digits
    allowed = [*string.ascii_lowercase, *string.digits]
    
    # Remove non-alphanumeric characters
    s_fixed = ''
    for letter in s:
        if letter in allowed:
            s_fixed += letter
            
    s_reversed = ''
    # For every letter
    for letter in s_fixed:
        # Reversed string = letter + reversed string
        s_reversed = letter + s_reversed
        
    # Check for equality
    if s_fixed == s_reversed:
        return True
    return False


# Test cases
for word in ['Bob', '**Bob****', 'Appsilon', 'A man, a plan, a canal: Panama']:
    print(f"Is {word} a palindrome? {is_palindrome(s=word)}")
    
    
    
    






Quickselect is a selection algorithm to find the k-th smallest element in an unordered list. It is related to the quick sort sorting algorithm.
Examples: 

Input: arr[] = {7, 10, 4, 3, 20, 15}
           k = 3
Output: 7

Input: arr[] = {7, 10, 4, 3, 20, 15}
           k = 4
Output: 10



# Python3 program of Quick Select
 
# Standard partition process of QuickSort().
# It considers the last element as pivot
# and moves all smaller element to left of
# it and greater elements to right
def partition(arr, l, r):
     
    x = arr[r]
    i = l
    for j in range(l, r):
         
        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
             
    arr[i], arr[r] = arr[r], arr[i]
    return i
 
# finds the kth position (of the sorted array)
# in a given unsorted array i.e this function
# can be used to find both kth largest and
# kth smallest element in the array.
# ASSUMPTION: all elements in arr[] are distinct
def kthSmallest(arr, l, r, k):
 
    # if k is smaller than number of
    # elements in array
    if (k > 0 and k <= r - l + 1):
 
        # Partition the array around last
        # element and get position of pivot
        # element in sorted array
        index = partition(arr, l, r)
 
        # if position is same as k
        if (index - l == k - 1):
            return arr[index]
 
        # If position is more, recur
        # for left subarray
        if (index - l > k - 1):
            return kthSmallest(arr, l, index - 1, k)
 
        # Else recur for right subarray
        return kthSmallest(arr, index + 1, r,
                            k - index + l - 1)
    print("Index out of bound")
 
# Driver Code
arr = [ 10, 4, 5, 8, 6, 11, 26 ]
n = len(arr)
k = 3
print("K-th smallest element is ", end = "")
print(kthSmallest(arr, 0, n - 1, k))
 
# This code is contributed by Muskan Kalra.




#########
Find the contiguous subarray within an array (containing at least one number) which has the largest product.

Return an integer corresponding to the maximum product possible.
Example :

Input : [2, 3, -2, 4]
Return : 6 

Possible with [2, 3]




#### O(n2)
class Solution:
	# @param A : tuple of integers
	# @return an integer
	def maxProduct(self, A):
        def prod(v):
            v2 =1
            for vi in v :
                v2 = v2 * vi
            return v2    

        pmax = -9999999999999
        n = len(A)
        for i2 in range(0,n+1):
            for i1 in range(0,i2):
              pi   = prod(A[i1:i2])
              pmax = max(pi, pmax)
        return pmax



#### O(N), using Stack 
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        n = len(nums)
        
        # the answer will be sum{ Max(subarray) - Min(subarray) } over all possible subarray
        # which decomposes to sum{Max(subarray)} - sum{Min(subarray)} over all possible subarray
        # so totalsum = maxsum - minsum
        # we calculate minsum and maxsum in two different loops
        minsum = maxsum = 0
        
        # first calculate sum{ Min(subarray) } over all subarrays
        # sum{ Min(subarray) } = sum(f(i) * nums[i]) ; i=0..n-1
        # where f(i) is number of subarrays where nums[i] is the minimum value
        # f(i) = (i - index of the previous smaller value) * (index of the next smaller value - i) * nums[i]
        # we can claculate these indices in linear time using a monotonically increasing stack.
        stack = []
        for next_smaller in range(n + 1):
			# we pop from the stack in order to satisfy the monotonically increasing order property
			# if we reach the end of the iteration and there are elements present in the stack, we pop all of them
            while stack and (next_smaller == n or nums[stack[-1]] > nums[next_smaller]):
                i = stack.pop()
                prev_smaller = stack[-1] if stack else -1
                minsum += nums[i] * (next_smaller - i) * (i - prev_smaller)
            stack.append(next_smaller)
            
        # then calculate sum{ Max(subarray) } over all subarrays
        # sum{ Max(subarray) } = sum(f'(i) * nums[i]) ; i=0..n-1
        # where f'(i) is number of subarrays where nums[i] is the maximum value
        # f'(i) = (i - index of the previous larger value) - (index of the next larger value - i) * nums[i]
        # this time we use a monotonically decreasing stack.
        stack = []
        for next_larger in range(n + 1):
			# we pop from the stack in order to satisfy the monotonically decreasing order property
			# if we reach the end of the iteration and there are elements present in the stack, we pop all of them
            while stack and (next_larger == n or nums[stack[-1]] < nums[next_larger]):
                i = stack.pop()
                prev_larger = stack[-1] if stack else -1
                maxsum += nums[i] * (next_larger - i) * (i - prev_larger)
            stack.append(next_larger)
        
        return maxsum - minsum



















































































































































































        
    
    