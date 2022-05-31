
 

 

    

 
 
########################################################################################
######################################################################################## 
0. Find duplicates in an array 
    Easy Accuracy: 20.69% Submissions: 100k+ Points: 2
    Given an array a[] of size N which contains elements from 0 to N-1, you need to find all the elements occurring more than once in the given array.

Example 1:
    Input:
    N = 4
    a[] = {0,3,1,2}
    Output: -1
    Explanation: N=4 and all elements from 0
    to (N-1 = 3) are present in the given
    array. Therefore output is -1.
    Example 2:

    Input:
    N = 5
    a[] = {2,3,1,2,3}
    Output: 2 3 
    Explanation: 2 and 3 occur more than once
    in the given array.
Your Task:
Complete the function duplicates() which takes array a[] and n as input as parameters and returns a list of elements that occur more than once in the given array in sorted manner. If no such element is found, return list containing [-1]. 

    Expected Time Complexity: O(n).
    Expected Auxiliary Space: O(n).
    Note : The extra space is only for the array to be returned.
    Try and perform all operation withing the provided array. 

    Constraints:
    1 <= N <= 105
    0 <= A[i] <= N-1, for each valid i
 
 
 
 

def printRepeating(arr, n):

    # First check all the
        # values that are
    # present in an array
        # then go to that
    # values as indexes
        # and increment by
    # the size of array
    for i in range(0, n):
        index = arr[i] % n
        arr[index] += n

    # Now check which value
        # exists more
    # than once by dividing
        # with the size
    # of array
    for i in range(0, n):
        if (arr[i]/n) >= 2:
            print(i, end=" ")


# Driver code
arr = [1, 6, 3, 1, 3, 6, 6]
arr_size = len(arr)

print("The repeating elements are:")

# Function call
printRepeating(arr, arr_size)







  
  
  
  
  
##################################################################  
3. Longest Substring Without Repeating Characters
https://leetcode.com/problems/longest-substring-without-repeating-characters/solution/
Medium

23928

1065

Add to List

Share
Given a string s, find the length of the longest substring without repeating characters.

 

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
 

Constraints:

0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces.
  
  
  
  
  Approach 2: Sliding Window
  class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        chars = [0] * 128

        left = right = 0

        res = 0
        while right < len(s):
            r = s[right]
            chars[ord(r)] += 1

            while chars[ord(r)] > 1:
                l = s[left]
                chars[ord(l)] -= 1
                left += 1

            res = max(res, right - left + 1)

            right += 1
        return res
  
  
  
    
  
  
  
  
  
  #####################################################################################################################
  1. Two Sum
Approach 3: One-pass Hash Table
Algorithm

It turns out we can do it in one-pass. While we are iterating and inserting elements into the hash table, we also look back to check if current element's complement already exists in the hash table. If it exists, we have found a solution and return the indices immediately.
Add to List



Share
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:
      Input: nums = [2,7,11,15], target = 9
      Output: [0,1]
      Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].


Example 2:
      Input: nums = [3,2,4], target = 6
      Output: [1,2]
      Example 3:

      Input: nums = [3,3], target = 6
      Output: [0,1]


Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.



  class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i
  
  
  
  
  
  
  
  
  
  
  
49. Group Anagrams
Medium
https://leetcode.com/problems/group-anagrams/solution/
   Given an array of strings strs, group the anagrams together. You can return the answer in any order.

   An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

 

Example 1:
    Input: strs = ["eat","tea","tan","ate","nat","bat"]
    Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

  
Example 2:
    Input: strs = [""]
    Output: [[""]]

  
Example 3:
    Input: strs = ["a"]
    Output: [["a"]]
 

Constraints:
    1 <= strs.length <= 104
    0 <= strs[i].length <= 100
    strs[i] consists of lowercase English letters.




##### Approach 2: Categorize by Count
Intuition

Two strings are anagrams if and only if their character counts (respective number of occurrences of each character) are the same.
  
  class Solution:
    def groupAnagrams(strs):
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()

      
Complexity Analysis

    Time Complexity: O(NK)O(NK), where NN is the length of strs, 
    and KK is the maximum length of a string in strs. 
    Counting each string is linear in the size of the string, and we count every string.

    Space Complexity: O(NK)O(NK), the total information content stored in ans.
  
  
  
  
  
  
  

  
    
    
#################################################################################### 
####################################################################################     
#### 844. Backspace String Compare
Given two strings s and t, return true if they are equal when both are typed into empty text editors. 
'#' means a backspace character.
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






 
 
#################################################################################### 
#################################################################################### 
 Minimum sum 
  https://www.geeksforgeeks.org/minimum-sum-two-numbers-formed-digits-array/

        Medium Accuracy: 48.73% Submissions: 32500 Points: 4
        Given an array Arr of size N such that each element is from the range 0 to 9. 
        Find the minimum possible sum of two numbers formed using the elements of the array. 
            All digits in the given array must be used to form the two numbers.


Example 1:
      Input:
      N = 6
      Arr[] = {6, 8, 4, 5, 2, 3}
      Output: 604
      Explanation: The minimum sum is formed 
      by numbers 358 and 246.


Example 2:
      Input:
      N = 5
      Arr[] = {5, 3, 0, 7, 4}
      Output: 82
      Explanation: The minimum sum is 
      formed by numbers 35 and 047.

  Your Task:
  You don't need to read input or print anything. Your task is to complete the function solve() which takes arr[] and n as input parameters and returns the minimum possible sum. As the number can be large, return string presentation of the number without leading zeroes.
 

    Expected Time Complexity: O(N*logN) Expected Auxiliary Space: O(1)

    Constraints:
    1 ≤ N ≤ 107
    0 ≤ Arri ≤ 9


# Function to find and return minimum sum of # two numbers formed from digits of the array.
def solve(arr, n):
    # sort the array
    arr.sort()
 
    # let two numbers be a and b
    a = 0; b = 0
    for i in range(n):
     
        # Fill a and b with every alternate
        # digit of input array
        if (i % 2 != 0):
            a = a * 10 + arr[i]
        else:
            b = b * 10 + arr[i]
 
    # return the sum
    return a + b
 
# Driver code
arr = [6, 8, 4, 5, 2, 3]
n = len(arr)
print("Sum is ", solve(arr, n))
 

 
    
    
    

















