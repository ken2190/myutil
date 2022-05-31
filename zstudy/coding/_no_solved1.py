


##############################################################################################
##############################################################################################
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











  
##############################################################################################
##############################################################################################
539. Minimum Time Difference
Medium
Share
Given a list of 24-hour clock time points in "HH:MM" format, return the minimum minutes difference between any two time-points in the list.
 

Example 1:

      Input: timePoints = ["23:59","00:00"]
      Output: 1


Example 2:

      Input: timePoints = ["00:00","23:59","00:00"]
      Output: 0
      

Constraints:

2 <= timePoints.length <= 2 * 104
timePoints[i] is in the format "HH:MM".











  
  
##############################################################################################
##############################################################################################
Sum of Middle Elements of two sorted arrays   : Not solved
        Medium Accuracy: 61.07% Submissions: 7817 Points: 4
        Given 2 sorted arrays Ar1 and Ar2 of size N each. Merge the given arrays and find the sum of the two middle elements of the merged array.

        
        Example 1:

        Input:
        N = 5
        Ar1[] = {1, 2, 4, 6, 10}
        Ar2[] = {4, 5, 6, 9, 12}
        Output: 11
        Explanation: The merged array looks like
        {1,2,4,4,5,6,6,9,10,12}. Sum of middle
        elements is 11 (5 + 6).
        

        Example 2:

        Input:
        N = 5
        Ar1[] = {1, 12, 15, 26, 38}
        Ar2[] = {2, 13, 17, 30, 45}
        Output: 32
        Explanation: The merged array looks like
        {1, 2, 12, 13, 15, 17, 26, 30, 38, 45} 
        sum of middle elements is 32 (15 + 17).
        

        Your Task:
        You don't need to read input or print anything. Your task is to complete the function findMidSum() which takes  ar1, ar2 and n as input parameters and returns the sum of middle elements. 

        

        Expected Time Complexity: O(log N)
        Expected Auxiliary Space: O(1)

        

        Constraints:
        1 <= N <= 103
        1 <= Ar1[i] <= 106
        1 <= Ar2[i] <= 106










##############################################################################################
##############################################################################################
Generate IP Addresses 
    Medium Accuracy: 43.42% Submissions: 14933 Points: 4
    Given a string S containing only digits, Your task is to complete the function genIp() which returns a vector containing 
    all possible combinations of valid IPv4 IP addresses and takes only a string S as its only argument.
    Note: Order doesn't matter.

 
  For string 11211 the IP address possible are 
  1.1.2.11
  1.1.21.1
  1.12.1.1
  11.2.1.1

Example 1:
    Input:
    S = 1111
    Output: 1.1.1.1
 
 
Example 2:
    Input:
    S = 55
    Output: -1
Expected Time Complexity: O(N * N * N * N)  O(N4)
Expected Auxiliary Space: O(N * N * N * N)

 
 Idea:  4 dots
    recursion.,

  one dot :
     [0,...N-4]
 
  two dot:
       [0...i1]
         [i1...n-3]
 
  thre dot:
        [0...i1]  , [i1...i2] , [i2..n - 2]
 
   ######## O(N)
   for i1 in range(0,n-4):     ###N En index
     for i2 in range(i1,n-3): 
       for i3 in range(i2, n-2): 
 
            ip = sent[0:i1] + '.' +  sent[i1:i2] + '.' + sent[i2:i3] + sent[i3:]   ### O(N)
               
            if isok(sent[0:i1]) and isok(sent[i1:i2]) and isok(sent[i2:i3]) and isok(sent[i3:in]) :  O(1) convetion to int, comparison.
                count += 1
 
            #### Validation if ip is ok 

 
 
# Python3 code to check valid possible IP
# Function checks whether IP digits
# are valid or not.
def is_valid(ip):
 
    # Splitting by "."
    ip = ip.split(".")
     
    # Checking for the corner cases
    for i in ip:
        if (len(i) > 3 or int(i) < 0 or
                          int(i) > 255):
            return False
        if len(i) > 1 and int(i) == 0:
            return False
        if (len(i) > 1 and int(i) != 0 and
            i[0] == '0'):
            return False
             
    return True
 
# Function converts string to IP address
def convert(s):
    sz = len(s)
 
    # Check for string size
    if sz > 12:
        return []
    snew = s
    l = []
 
    # Generating different combinations.
    for i in range(1, sz - 2):
        for j in range(i + 1, sz - 1):
            for k in range(j + 1, sz):
                snew = snew[:k] + "." + snew[k:]
                snew = snew[:j] + "." + snew[j:]
                snew = snew[:i] + "." + snew[i:]
                 
                # Check for the validity of combination
                if is_valid(snew):  ### O(N) validation
                    l.append(snew)
                     
                snew = s
                 
    return l

# Driver code        
A = "25525511135"
B = "25505011535"



