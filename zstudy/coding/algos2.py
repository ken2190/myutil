

##############################################################################################
##############################################################################################
Swapping pairs make sum equal 
  Easy Accuracy: 41.35% Submissions: 19608 Points: 2
  Given two arrays of integers A[] and B[] of size N and M, the task is to check 
  if a pair of values (one value from each array) exists such that swapping the elements 
  of the pair will make the sum of two arrays equal.

 

Example 1:
    Input: N = 6, M = 4
    A[] = {4, 1, 2, 1, 1, 2}
    B[] = (3, 6, 3, 3)
    
    Output: 1
    Explanation: Sum of elements in A[] = 11
    Sum of elements in B[] = 15, To get same 
    sum from both arrays, we can swap following 
    values: 1 from A[] and 3 from B[]

      
Example 2:

    Input: N = 4, M = 4
    A[] = {5, 7, 4, 6}
    B[] = {1, 2, 3, 8}
    Output: 1
    Explanation: We can swap 6 from array 
    A[] and 2 from array B[]

Should I write code ? or explain ?
for this one, we can write code here


Method 4 (Hashing) 
     We can solve this problem in O(m+n) time and O(m) auxiliary space. 

python code:
  
  m = len(array1)
  n = len(array2)
  
  # Find sum1 and sum2   
  sum1 = sum(array1)
  sum2 = sum(array2)
  
  # Now, lets make the hashset for array1
  set1 = {}
  for x in array1:
    set1.insert(x)
  
  # The key idea :
  # We want to chose elements in array1 and array2 such that their difference is (sum1-sum2)/2
  # Swapping them would make the array sums equal
  ## are you here ????
  
  diff= (sum1-sum2)/2
  for i in range(0, n):
    
    ### dont understand this condition, 
    ### Cool let me explain it 
    if array2[i]+diff in set1:
      print(array2[i]+diff,  array2[i]  )
      flag = True
      break
  
	### The main idea is to find 2 elements say x in array1 and y in array2 such that swapping would make sums equal
    ### sum1_new = sum1 + y - x
    ### sum2_new = sum2 + x - y
    
    ### We need the sum1_new and sum2_new to be equal right so,
    ### sum1_new 

     OK, I understand NOW, BUT how to get this idea ???
      Is it a standard pattern ?
      
    ### Say sum1 > sum2 , intutively u would want to chose a larger element from array1 and a smaller from array2 and swap right ?

    
  x1-y2 = (sum1-sum2)/2
  
  sum1_new = sum1 -x1 + y2 = sum1 - (sum1-sum2)/2
  sum2_new = sum2 -y2 + x1 = sum2 + (sum1-sum2)/2
   
  ### it';s just the  math condition  ( Right )
  sum1_new = sum2_new
  sum1 -x1 + y2  =  sum2 -y2 + x1
  sum1-sum2 =    2*(x1-y2)
  x1-y2 = (sum1-sum2)/2
  
  for x1 in array1:
    if x1+ diff0 in set_array2 :
      then OK works.   ## hash == O(1)  
  
     ### USe a hash to have O(1)
   No need to say stupid remark on hash, 
    oviously we dont build the hash map n times.... (cool)
    
    This OK, it's directly yhe MATH condition + Hash

  Correct !!!
    OK, lets move to next problem and skip the stupid solution.
  ( The majority element one right ??? )
  
  
  ### Invariant  is  (sum1-sum2)/2
  ####
  
yes, we just implement the description below  
// assume array1 is small i.e. (m < n)
// where m is array1.length and n is array2.length

1. Find sum1(sum of small array elements) ans sum2

  (sum of larger array elements). // time O(m+n)

2. Make a hashset for small array(here array1).

3. Calculate diff as (sum1-sum2)/2.

4. Run a loop for array2

     for (int i equal to 0 to n-1)

       if (hashset contains (array2[i]+diff))

           print array2[i]+diff and array2[i]

           set flag  and break;

5. If flag is unset then there is no such kind of 

pair.






##############################################################################################
##############################################################################################
Majority Element 
    Medium Accuracy: 48.6% Submissions: 100k+ Points: 4
    Given an array A of N elements. Find the majority element in the array. 
    A majority element in an array A of size N is 
    an element that appears more than N/2 times in the array.
 
Example 1:
    Input:
    N = 3 
    A[] = {1,2,3} 
    Output:
    -1
    Explanation:
    Since, each element in 
    {1,2,3} appears only once so there 
    is no majority element.
    
  
Example 2:
    Input:
    N = 5 
    A[] = {3,1,3,3,2} 
    Output:
    3
    Explanation:
    Since, 3 is present more
    than N/2 times, so it is 
    the majority element


  METHOD 4 (Using Hashmap): 


Approach: This method is somewhat similar to Moore voting algorithm 
  in terms of time complexity, but in this case,
  there is no need for the second step of Moore voting algorithm. 
  But as usual, here space complexity becomes O(n). 
In Hashmap(key-value pair), at value, maintain a count for each element(key) and whenever the count is greater than half of the array length, return that key(majority element). 
 
  
Algorithm:
    Create a hashmap to store a key-value pair, i.e. element-frequency pair.
    Traverse the array from start to end.
    For every element in the array, insert the element in the hashmap 
    if the element does not exist as key, else fetch the value of the key ( array[i] ), 
    and increase the value by 1
    
    If the count is greater than half then print the majority element and break.
    If no majority element is found print "No Majority element"

    is very easy
    
    ddict = {}
    for x in range(array1):
      if x not in ddict : ddict[x]=1
      else: ddict[x] +=1 
    
    for key,val in ddict.items():
      if val > n//2 : return key
    return -1
  
  no, it was vrey easy, usually hashmap are very easy.
  
  let move to next ( COOL )
      ( Yeah this is correct )
    ( So is there any doubt ??? )
Should I use hashmaps or the moore voting algo ???

hashmap is MORE generic solution, 

Cool will explain that just that moore is better in terms of space ( Its O(1))

ctr_map = {}
for x in array:
  if x not in ctr_map:
    ctr_map[x] = 1
  else:
    ctr_map[x] += 1

# Python3 program for finding out majority 
# element in an array 

def findMajority(arr, size):
    m = {}
    for i in range(size):
        if arr[i] in m:
            m[arr[i]] += 1
        else:
            m[arr[i]] = 1
    count = 0
    for key in m:
        if m[key] > size / 2:
            count = 1
            print("Majority found :-",key)
            break
    if(count == 0):
        print("No Majority element")

# Driver code 
arr = [2, 2, 2, 2, 5, 5, 2, 3, 3] 
n = len(arr)

# Function calling 
findMajority(arr, n)

# This code is contributed by ankush_953





##############################################################################################
##############################################################################################

Find Missing And Repeating 
Medium Accuracy: 37.77% Submissions: 100k+ Points: 4
Given an unsorted array Arr of size N of positive integers.
One number 'A' from set {1, 2, …N} is missing and one number 'B' occurs twice in array. 
Find these two numbers.

Example 1:
    Input:
    N = 2
    Arr[] = {2, 2}
    Output: 2 1
    Explanation: Repeating number is 2 and 
    smallest positive missing number is 1.

Example 2:
    Input:
    N = 3
    Arr[] = {1, 3, 3}
    Output: 3 2
    Explanation: Repeating number is 3 and 
    smallest positive missing number is 2.

Expected Time Complexity: O(N)
Expected Auxiliary Space: O(1)




################################################################################
Method 6 (Use a Map)
    Approach: 
    This method involves creating a Hashtable with the help of Map. 
    In this, the elements are mapped to their natural index. 
    In this process, if an element is mapped twice, then it is the repeating element. 
    And if an element's mapping is not there, then it is the missing element.

n = len(array)
ctr_map = {}
for x in array:
  if x not in ctr_map:
    ctr_map[x] = 1
  else:
    ctr_map[x] += 1
  
missing_ele, duplicate_ele = -1, -1
for key in range(1,n+1):
  if(key not in ctr_map):
    missing_ele = key
  elif(ctr_map[key]==2):
    duplicate_ele = key
  else:
    pass

### Is this fine ???
yes, ahashmap is trivial, direcy solution

### Should I explain an alternate with O(1) space ???
No need,
lets do below







##############################################################################################
##############################################################################################
Maximum Index 
    Medium Accuracy: 42.72% Submissions: 81531 Points: 4
    Given an array A[] of N positive integers. 
    The task is to find the maximum of j - i subjected to the constraint of A[i] < A[j] and i < j.


Example 1:
    Input:
    N = 2
    A[] = {1, 10}
    Output:
    1
    Explanation:
    A[0]<A[1] so (j-i) is 1-0 = 1.

    
Example 2:
    Input:
    N = 9
    A[] = {34, 8, 10, 3, 2, 80, 30, 33, 1}
    Output:
    6
    Explanation:
    In the given array A[1] < A[7]
    satisfying the required 
    condition(A[i] < A[j]) thus giving 
    the maximum difference of j - i 
    which is 6(7-1).


    
#### Binary Search
can you exa
# For a given array arr,  calculates the maximum j – i  such that arr[j] > arr[i] 

v = [34, 8, 10, 3, 2, 80, 30, 33, 1];
n = len(v);
maxFromEnd = [-38749432] * (n + 1);

#### what is it ????
# Create an array maxfromEnd
for i in range(n - 1, 0, -1):
    maxFromEnd[i] = max(maxFromEnd[i + 1], 
                        v[i]);
    
result = 0;
for i in range(0, n):
    low  = i + 1; 
    high = n - 1; 
    ans  = i;

    ### Binary search
    while (low <= high):
        mid = int((low + high) / 2);

        if (v[i] <= maxFromEnd[mid]):

            # We store this as currentanswer and look for further larger number to the right side
            ans = max(ans, mid);
            low = mid + 1;
        else:
            high = mid - 1;        

    # Keeping a track of the maximum difference in indices
    result = max(result, ans - i);

print(result, end = "");
    


  

    
    
# Python3 program to find the maximum j – i such that arr[j] > arr[i] For a given array arr[], returns
# the maximum j – i such that arr[j] > arr[i]

### N**2 , stupid solution
def maxIndexDiff(arr, n):
    maxDiff = -1
    for i in range(0, n):
        j = n - 1
        while(j > i):
            if arr[j] > arr[i] and maxDiff < (j - i):  ### update the max
                maxDiff = j - i
            j -= 1

    return maxDiff

arr = [9, 2, 3, 4, 5, 6, 7, 8, 18, 0]
n = len(arr)
maxDiff = maxIndexDiff(arr, n)
print(maxDiff)












































































##############################################################################################
##############################################################################################
Median in a stream of integers (running integers)
      Difficulty Level : Hard
      Last Updated : 28 Mar, 2022
https://www.geeksforgeeks.org/median-of-stream-of-integers-running-integers/
          
        
      Given that integers are read from a data stream. Find median of elements read so 
      for in an efficient way. For simplicity assume, there are no duplicates. 
      For example, let us consider the stream 5, 15, 1, 3 … 


      After reading 1st element of stream - 5 -> median - 5
      After reading 2nd element of stream - 5, 15 -> median - 10
      After reading 3rd element of stream - 5, 15, 1 -> median - 5
      After reading 4th element of stream - 5, 15, 1, 3 -> median - 4, so on...
      Making it clear, when the input size is odd, we take the middle element of sorted data. 
      If the input size is even, we pick the average of the middle two elements in the sorted stream.
      Note that output is the effective median of integers read from the stream so far. 
      Such an algorithm is called an online algorithm. 
      Any algorithm that can guarantee the output of i-elements after processing i-th element, 
      is said to be online algorithm. Let us discuss three solutions to the above problem.



  

Time Complexity: If we omit the way how stream was read, complexity of median finding is O(N log N), 
  as we need to read the stream, and due to heap insertions/deletions.

Auxiliary Space: O(N)
At first glance the above code may look complex. If you read the code carefully, it is simple algorithm.   

# Heap based
### Key idea :
    Maintain 2 heaps ( A left heap & aa right heap )

    Left Right 
# Condition : Left <= RIght & (Right - Left) <=1  
              Left = Right     (Even)
         OR   Left = Right-1   (Odd)

A= [5, 15, 1, 3, 2, 8, 7, 9, 10, 6, 11, 4]

Left -> [] Right -> [5]
Left -> [5] Right -> [15]
Left -> [1] Right -> [5,15] =>    ## why 5 is needed in [5,15]

Left -> [3] Right -> [  ] =>    ##

for i,x in enumerate(arr):
  	# IF i is even : 
  	if i%2 == 0:
      	# Intially size(Left) == size(Right)
        # Finally size(Left) == size(Right) - 1
        # So, u need to increase size of right by 1 
        if x < min(Left):
          	val = Left.min_pop()  ## remove the current min
            Left.insert(x)     ## new min
            Right.insert(val)  ## moving to the right
        else:
          	Right.insert(x)   ### keep the min in Left,  only change the right.
            
	else:
      	if x > max(Right):
          val = Right.max_pop()
          Right.insert(x)
          Left.insert(val)   ####val added to left, right has new max == x
        else:
          Left.insert(x)

          
#### Sliding window  based on position of X,           
    A[i1:i2]      and i1, i2 are moving....      

  Why left and right  ?  
  why not only  i1, i2 indexes ....
    We can only use index   i1, i2
    
    sometimes, there are problem with i1, i2 in sliding window....   A[i1:i2]  
       Inserting is log(n)
       --> total is n/log(n)      
      
    
    

def getMedian():
    if len(Left) != len(Right):
        return Right[0]
    else:
        return (Right[0] + Left[0])/2
           

        
### pseudo code is ok,

    
python : by default its a minheap,  so negative value for Maxheap.  --> - values for MaxHeap
    https://www.geeksforgeeks.org/heap-queue-or-heapq-in-python/  

def insertHeaps(num):
    heappush(maxHeap,-num)                ### Pushing negative element to obtain a minHeap for
    heappush(minHeap, -heappop(maxHeap))    ### the negative counterpart
   
   if len(minHeap) > len(maxHeap):
        heappush(maxHeap,-heappop(minHeap))



######## code
from heapq import heappush, heappop, heapify
import math
minHeap=[]
heapify(minHeap)

maxHeap=[]
heapify(maxHeap)

def insertHeaps(num):
    heappush(maxHeap,-num)                ### Pushing negative element to obtain a minHeap for
    heappush(minHeap,-heappop(maxHeap))    ### the negative counterpart
   
   if len(minHeap) > len(maxHeap):
        heappush(maxHeap,-heappop(minHeap))

        
def getMedian():
    if len(minHeap)!= len(maxHeap):
        return -maxHeap[0]
    else:
        return (minHeap[0]- maxHeap[0])/2
   

A= [5, 15, 1, 3, 2, 8, 7, 9, 10, 6, 11, 4]
n= len(A)
for i in range(n):
    insertHeaps(A[i])
    print(math.floor(getMedian()))  
  
  
  
  
  
  
  
  
O(N**2)  

##### Function to find position to insert current element of stream using binary search
def binarySearch(arr, item, low, high):
    if (low >= high):
        return (low + 1) if (item > arr[low]) else low

    mid = (low + high) // 2

    if (item == arr[mid]):
        return mid + 1

    if (item > arr[mid]):
        return binarySearch(arr, item, mid + 1, high)
    else :      
        return binarySearch(arr, item, low, mid - 1)

      
  
# Function to print median of stream of integers
def printMedian(arr, n):
    i, j, pos, num = 0, 0, 0, 0
    count = 1
    print(f"Median after reading 1 element is {arr[0]}")

    for i in range(1, n):
        median = 0
        j      = i - 1
        num    = arr[i]

        # find position to insert current element in sorted part of array
        pos = binarySearch(arr, num, 0, j)

        # move elements to right to create space to insert the current element
        while (j >= pos):
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = num

        # increment count of sorted elements in array
        count += 1

        # if odd number of integers are read from stream
        # then middle element in sorted order is median
        # else average of middle elements is median
        if (count % 2 != 0):
            median = arr[count // 2]

        else:
            median = (arr[(count // 2) - 1] + arr[count // 2]) // 2

        print(f"Median after reading {i + 1} elements is {median} ")

        
#### Test        
arr = [5, 15, 1, 3, 2, 8, 7, 9, 10, 6, 11, 4]
n = len(arr)
printMedian(arr, n)









##############################################################################################
##############################################################################################
Maximum Product Subarray 
    Medium Accuracy: 29.84% Submissions: 100k+ Points: 4
    Given an array Arr[] that contains N integers (may be positive, negative or zero). 
    Find the product of the maximum product subarray.

Example 1:

      Input:
      N = 5
      Arr[] = {6, -3, -10, 0, 2}
      Output: 180
      Explanation: Subarray with maximum product
      is [6, -3, -10] which gives product as 180.
      
      
      
Example 2:
    Input:
    N = 6
    Arr[] = {2, 3, 4, 5, -1, 0}
    Output: 120
    Explanation: Subarray with maximum product
    is [2, 3, 4, 5] which gives product as 120.

Expected Time Complexity: O(N)
Expected Auxiliary Space: O(1)

Constraints:
1 ≤ N ≤ 500
-102 ≤ Arri ≤ 102


sliding window with i1, i2


[a_0,a_1,a_2,....]

results =
for i in range(n):
  # Find max product of a subarray ending at i
  result = max()

you can go below ,


#  OPtimal Function to find maximum  product subarray
def maxProduct(arr, n):
    
    # Variables to store maximum and  minimum product till ith index.
    minVal = arr[0]
    maxVal = arr[0]
    maxProduct = arr[0]

    for i in range(1, n, 1):
        
        # When multiplied by negative number,maxVal becomes minVal
        # and minVal becomes maxVal.
        """
        if (arr[i] < 0):   ### Reverse  min <--> max
            temp = maxVal
            maxVal = minVal
            minVal = temp
            
        # maxVal and minVal stores the  product of subarray ending at arr[i].
        maxVal = max(arr[i], maxVal * arr[i])
        minVal = min(arr[i], minVal * arr[i])
        """
        
        ""
        ### to handl negative values.... 
        maxVal = max(a[i], max(a[i]*minVal,  a[i]*maxVal)  )
        minVal = min(a[i], min(a[i]*minVal,  a[i]*maxVal) )
        
        # Max Product of array.
        maxProduct = max(maxProduct, maxVal)

    # Return maximum product 
    # found in array.
    return maxProduct


Exanmple  
    [0,  1,  3]   -->    3 
  
     max(1, 0*....)  =1 ,    ### reset of the product at each step, more 
    
    
 Instewad of product, Max sum  --->  we just to keep  MaxVal ONLY.

   teshnically
      current maxVal vs a[i]   (intermediate maxVal  to handle issues/special case of negative, what ever things we need to manage.... )  
      Global_MaxVal
  
    Sliding window way with only one step, 
    Good pattern.
    

arr = [-1, -3, -10, 0, 60]

n = len(arr)

print("Maximum Subarray product is",
                 maxProduct(arr, n))






###### O(n2)

# Python3 program to find Maximum Product Subarray
# Returns the product of max product subarray.
def maxSubarrayProduct(arr, n):
    # Initializing result
    result = arr[0]
 
    for i in range(n):     
        mul = arr[i]
       
        # traversing in current subarray
        for j in range(i + 1, n):
         
            # updating result every time
            # to keep an eye over the maximum product
            result = max(result, mul)
            mul *= arr[j]
         
        # updating the result for (n-1)th index.
        result = max(result, mul)
     
    return result
 
# Driver code
arr = [ 1, -2, -3, 0, 7, -8, -2 ]
n = len(arr)
print("Maximum Sub array product is" , maxSubarrayProduct(arr, n))
 
# This code is contributed by divyeshrabadiya07



The idea is to traverse array from left to right keeping two variables minVal and maxVal which represents the minimum and maximum product value till the ith index of the array. Now, if the ith element of the array is negative that means now the values of minVal and maxVal will be swapped as value of maxVal will become minimum by multiplying it with a negative number. Now, compare the minVal and maxVal. 
The value of minVal and maxVal depends on the current index element or the product of the current index element and the previous minVal and maxVal respectively.
Below is the implementation of above approach: 
 








##############################################################################################
##############################################################################################
Triplet Sum in Array 
    Medium Accuracy: 49.0% Submissions: 100k+ Points: 4
    Given an array arr of size n and an integer X. Find if there's a triplet in the array which sums up to the given integer X.


Example 1:

    Input:
    n = 6, X = 13
    arr[] = [1 4 45 6 10 8]
    Output:
    1
    Explanation:
    The triplet {1, 4, 8} in 
    the array sums up to 13.

    
Example 2:

    Input:
    n = 5, X = 10
    arr[] = [1 2 4 3 6]
    Output:
    1
    Explanation:
    The triplet {1, 3, 6} in 
    the array sums up to 10.

    Expected Time Complexity: O(n2)
    Expected Auxiliary Space: O(1)

#### Best is hash map on the last value.      
   a[i], a[j], a[k]  in A  and a[k]   = X

  for i in range(n):
    for j in range(i+1,n):  ### Sliding
      # X - a[i] - a[j] is present in the array using Has in O(1)
  
  4 numbers , total sum = X   ---> is it in O(N3)  ???/
      is it a DP ????
  https://leetcode.com/problems/4sum/

    Time Complexity: O(n^{k - 1})O(n k−1 ), or O(n^3)O(n 3
 ) for 4Sum. We have k - 2k−2 loops, and twoSum is O(n)O(n).  
  
  Hash is much better/simpler.
  
  
  

  # returns true if there is triplet
# with sum equal to 'sum' present
# in A[]. Also, prints the triplet
def find3Numbers(A, arr_size, sum):

    # Sort the elements 
    A.sort()

    # Now fix the first element 
    # one by one and find the
    # other two elements 
    for i in range(0, arr_size-2):
    

        # To find the other two elements,
        # start two index variables from
        # two corners of the array and
        # move them toward each other
        
        # index of the first element
        # in the remaining elements
        l = i + 1 
        
        # index of the last element
        r = arr_size-1 
        while (l < r):
        
            if( A[i] + A[l] + A[r] == sum):
                print("Triplet is", A[i], 
                     ', ', A[l], ', ', A[r]);
                return True
            
            elif (A[i] + A[l] + A[r] < sum):
                l += 1
            else: # A[i] + A[l] + A[r] > sum
                r -= 1

    # If we reach here, then
    # no triplet was found
    return False


  

A = [1, 4, 45, 6, 10, 8]
sum = 22
arr_size = len(A)

find3Numbers(A, arr_size, sum)


  
  
  
  
# Python3 program to find a triplet 
# that sum to a given value

# returns true if there is triplet with
# sum equal to 'sum' present in A[]. 
# Also, prints the triplet
def find3Numbers(A, arr_size, sum):

    # Fix the first element as A[i]
    for i in range( 0, arr_size-2):

        # Fix the second element as A[j]
        for j in range(i + 1, arr_size-1): 
            
            # Now look for the third number
            for k in range(j + 1, arr_size):
                if A[i] + A[j] + A[k] == sum:
                    print("Triplet is", A[i],
                          ", ", A[j], ", ", A[k])
                    return True
    
    # If we reach here, then no 
    # triplet was found
    return False

# Driver program to test above function 
A = [1, 4, 45, 6, 10, 8]
sum = 22
arr_size = len(A)
  
  
  
  
  
  
  
  
  
  
##############################################################################################
##############################################################################################

Sum of Middle Elements of two sorted arrays 
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
Smallest Positive missing number  
    Medium Accuracy: 45.09% Submissions: 100k+ Points: 4
    You are given an array arr[] of N integers including 0. 
    The task is to find the smallest positive number missing from the array.

     missing.
  mex of an array 
  mex -> is the smalles +ve number that is not present in the array 

Example 1:

    Input:
    N = 5
    arr[] = {1,2,3,4,5}
    Output: 6      why 6 is missing ????
    Explanation: Smallest positive missing 
    number is 6.
    
    
Example 2:
    Input:
    N = 5
    arr[] = {0,-10,1,3,-20}
    Output: 2      3 is max
    Explanation: Smallest positive missing 
    number is 2.
    Your Task:
    The task is to complete the function missingNumber() which returns the smallest positive missing number in the array.

    Expected Time Complexity: O(N).
    Expected Auxiliary Space: O(1).

Constraints:
1 <= N <= 106
-106 <= arr[i] <= 106


###### Soliution 2, using Hash key
https://leetcode.com/problems/first-missing-positive/solution/
##ok 

### Distribution of val : frequency,   
## we check the one missing.
### checking any side conditions, if it is ok.

def findMissingPositive(arr: list[int]) -> int: 
  maxint =107
  
  hash_list = [0] * maxint    ### all values, dict
  
  for element in arr: 
    if element <= 0:
      continue
  
    #### element < 107
    hash_list[element] += 1 ## increase frequency
  
  ans = 1 
  
  while True: 
    if hash_list[ans] == 0:  ## missing 
      return ans ####
    
    ans += 1 

    





##############################################################################################
##############################################################################################
Count Inversions 
    Medium Accuracy: 39.43% Submissions: 100k+ Points: 4
    Given an array of integers. Find the Inversion Count in the array. 

    Inversion Count: For an array, inversion count indicates how far (or close) the array is from being sorted.
    If array is already sorted then the inversion count is 0. 
    If an array is sorted in the reverse order then the inversion count is the maximum. 
    
    ####  Not merge sort invresion
    Formally, two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j.


Example 1:
    Input: N = 5, arr[] = {2, 4, 1, 3, 5}
    Output: 3
    Explanation: The sequence 2, 4, 1, 3, 5 
    has three inversions (2, 1), (4, 1), (4, 3).

Example 2:
    Input: N = 5
    arr[] = {2, 3, 4, 5, 6}
    Output: 0
    Explanation: As the sequence is already 
    sorted so there is no inversion count.

Example 3:
    Input: N = 3, arr[] = {10, 10, 10}
    Output: 0
    Explanation: As all the elements of array 
    are same, so there is no inversion count.

Expected Time Complexity: O(NLogN).
Expected Auxiliary Space: O(N).


#######  
O(N2), Naive solution
# Python3 program to count inversions in an array
def getInvCount(arr, n):

    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] > arr[j]): ###  parse in i,j the array if elements are sorted.
                inv_count += 1

    return inv_count

arr = [1, 20, 6, 4, 5]
n = len(arr)
print("Number of inversions are",      getInvCount(arr, n))



###
def solution(array: list[int], N: int) -> int: 
  
  
  array_element_indices = {} 
  
  sorted_array = array
  sorted_array.sort()    ### N log_N
  
  index = 0 
  for array_element in array: 
    array_element_indices[array_element] = index 
    index += 1 
  
  
  inversion_count = 0 
  for i in range(n): 
    
    #### your Condition   <-->   two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j.
    #### Comparison current array state with Global Sorted,
    #### if elt_i not sorted --> permute to have it sorted. and count as one inversion.
    ### Sort 1st, and just follow the sorted values.
    if array[i] != sorted_array[i]: 
      inversion_count += 1 
      
      swap(array[i] , array[ array_element_indices[sorted_array[i]]  ]  )

      ioriginal = array_element_indices[ sorted_array[i] ] 
      tmp = array[i]
      array[i] = array[ ioriginal  ]  ####  sorted value 
      array[ ioriginal  ]  = tmp
        
  return inversion_count
  
#### More genneric pattern
  1) we sort to get correct state/
  2) compare correct/final state with initial state 
     and iterate to count the wrong item.
    
    MergeSort couting <> than the condition couting, Be careful !!!!!
    
    
  
  
  
array [ 7,3,2]
array_drot  [ 2,3,7 ]

array_indices = {
  7: 0, 
  3: 1, 
  2: 2, 
}

array = [2 , 3 , 7]
array_sorted = []
---> 2 investions

Final step: array is sorted.
### a sort step by step  with comparisng

  
 



New URL for zoom , sorry , please reconnect using below
Join Zoom Meeting
https://us05web.zoom.us/j/2933746463?pwd=WUhRWkx0NWNZRVBFVjZ4enV6Y1R2QT09






######
https://leetcode.com/problems/global-and-local-inversions/solution/

  
  
  
  
  

##############################################################################################
##############################################################################################
Subarray with given sum 
    Easy Accuracy: 39.71% Submissions: 100k+ Points: 2
    Given an unsorted array A of size N that contains only non-negative integers, 
    find a continuous sub-array which adds to a given number S.

    In case of multiple subarrays, return the subarray which comes first on moving from left to right.
    Empty [] 

 

Example 1:
    Input:
    N = 5, S = 12
    A[] = {1,2,3,7,5}
    Output: 2 4
    Explanation: The sum of elements 
    from 2nd position to 4th position 
    is 12.


Example 2:
    Input:
    N = 10, S = 15
    A[] = {1,2,3,4,5,6,7,8,9,10}
    Output: 1 5
    Explanation: The sum of elements 
    from 1st position to 5th position
    is 15.

    O(N) complexity
    
##### Write down the idea    
  window array, to keep track on going sum
  and check with target sum.
  
  while i < n :  ### while not finish the array  
    
    ### sliding window
    while curr_sum > taget_sum and istart <= i :  ## can remove all of them.
        ### Decrease size of window array.
        remove 1st element at  istart
        increase start of current array
        
    curr_sum =+  arr[i]     
    i = i + 1
        
    if curr_sum = target_sum  : exit
    
    thats ok for today.
    no need to do more. thanks you.
    
    2-3 times as week , we can adujust schedule., no worries
    thaknk yoyu vem
    
Still in O[N]    
    
    curr_sum =
    
    Good problems today
    
    
    
    
# Returns true if the 
# there is a subarray 
# of arr[] with sum
# equal to 'sum' 
# otherwise returns 
# false. Also, prints 
# the result.
def subArraySum(arr, n, sum_):
    
    # Initialize curr_sum as
    # value of first element
    # and starting point as 0 
    curr_sum = arr[0]
    start = 0

    # Add elements one by 
    # one to curr_sum and 
    # if the curr_sum exceeds 
    # the sum, then remove 
    # starting element 
    i = 1
    while i <= n:
        
        # If curr_sum exceeds
        # the sum, then remove
        # the starting elements
        while curr_sum > sum_ and start < i-1:
        
            curr_sum = curr_sum - arr[start]
            start += 1
            
        # If curr_sum becomes
        # equal to sum, then
        # return true
        if curr_sum == sum_:
            print ("Sum found between indexes")
            print ("% d and % d"%(start, i-1))
            return 1

        # Add this element 
        # to curr_sum
        if i < n:
            curr_sum = curr_sum + arr[i]
        i += 1

    # If we reach here, 
    # then no subarray
    print ("No subarray found")
    return 0

# Driver program
arr = [15, 2, 4, 8, 9, 5, 10, 23]
n = len(arr)
sum_ = 23

subArraySum(arr, n, sum_)

# This code is Contributed by shreyanshi_arun.







    
    
    
O(N2), naive solution
# equal to 'sum' 
# otherwise returns
# false. Also, prints
# the result 
def subArraySum(arr, n, sum_):
    
    # Pick a starting 
    # point
    for i in range(n):
        curr_sum = arr[i]
    
        # try all subarrays
        # starting with 'i'
        j = i + 1
        while j <= n:
        
            if curr_sum == sum_:
                print ("Sum found between")
                print("indexes % d and % d"%( i, j-1))
                
                return 1
                
            if curr_sum > sum_ or j == n:
                break
            
            curr_sum = curr_sum + arr[j]
            j += 1

    print ("No subarray found")
    return 0









##############################################################################################
##############################################################################################
Longest Prefix Suffix 
    Medium Accuracy: 49.39% Submissions: 41665 Points: 4
    Given a string of characters, find the length of the longest proper prefix which is also a proper suffix.

    NOTE: Prefix and suffix can be overlapping but 
      they should not be equal to the entire string.

Example 1:
  Input: s = "abab"
  Output: 2
  Explanation: "ab" is the longest proper 
  prefix and suffix. 

  
Example 2:
      Input: s = "aaaa"
      Output: 3
      Explanation: "aaa" is the longest proper 
      prefix and suffix. 

      
  Expected Time Complexity: O(|s|)
  Expected Auxiliary Space: O(|s|)

  Constraints:
  1 ≤ |s| ≤ 105
  s contains lower case English alphabets

  
  ##### O(N2)  this one :
  for i  in range(0, len(sent)) :
    
     if  sent[:i]  == sent[-i:] :   ###  Lenglitgh of i  x i 
  

  ####  O(n)
  for i  in range(0,  len(sent)  ) :
     if s[i] == s[-i] :
        pref += s[i]
        continue
        
     else :   
        break
        
   ###   they should not be equal to the entire string.
   if len(pref) == len(sent) :
      return len(pref[:-1])
   else :
       return len(pref)
  
  
  #### example
  aaaaaaa  --->  aaaaaa  : 5
  
  aa aa
  
  
  
  
# Efficient Python 3 programto find length of # the longest prefix 
# which is also suffix

# Returns length of the longest prefix
# which is also suffix and the two do
# not overlap. This function mainly is
# copy computeLPSArray() of in below post
# https://www.geeksforgeeks.org/searching-for-patterns-set-2-kmp-algorithm/
def longestPrefixSuffix(s) :
    n = len(s)
    lps = [0] * n   # lps[0] is always 0
 
    # length of the previous
    # longest prefix suffix
    l = 0 
    
    # the loop calculates lps[i]
    # for i = 1 to n-1
    i = 1 
    while (i < n) :
        if (s[i] == s[l]) :
            l = l + 1
            lps[i] = l
            i = i + 1
            ### KMP algorithm   Knuth–Morris–Pratt string-searching algorithm 
            """
            searches for occurrences of a "word" W within a main "text string" S 
            by employing the observation that when a mismatch occurs, 
            the word itself embodies sufficient information 
            to determine where the next match could begin, 
            thus bypassing re-examination of previously matched characters.
            
            """
        
        else :
            #### lps == l when match, otherwise 0 
            # (pat[i] != pat[len])
            # This is tricky. Consider the example. AAA CAA AA and i = 7. The idea is
            # similar to search step.  KMV algorithm
            if (l != 0) :
                l = lps[l-1] 
                # Also, note that we do
                # not increment i here
            
            else :
                # if (len == 0)
                lps[i] = 0
                i = i + 1
 
    res = lps[n-1]
 
    # Since we are looking for# non overlapping parts.
    if(res > n/2) :
        return n//2 
    else : 
        return res
        
        
###### Driver program to test above function
s = "abcab"
print(longestPrefixSuffix(s))




##############################################################################################
##############################################################################################
Parenthesis Checker 
    Easy Accuracy: 49.12% Submissions: 100k+ Points: 2
    Given an expression string x. Examine whether the pairs and the orders of “{“,”}”,”(“,”)”,”[“,”]” are correct in exp.
    For example, the function should return 'true' for exp = “[()]{}{[()()]()}” and 'false' for exp = “[(])”.

Example 1:
    Input:
    {([])}
    Output: 
    true
    Explanation: 
    { ( [ ] ) }. Same colored brackets can form 
    balaced pairs, with 0 number of 
    unbalanced bracket.


                                                                                                        
##### Need to use stack :  to push, pop ONLY the opening bracket.                                                                                                     
ddict = {'(':0}                                                                                                       
for c in sent:
                                                                                                        
    if c in "([{" 
      ll.push(c)
                                                                                                        
    elif:  ### this si closing bracket
       if len(ll) == 0: 
          return False 
                                                                                                        
       last_open = ll.pop()
       if compare(last_open, c)  :  ###  "("  == ")" 
          return False                                                                                                         
                                                                                                        
return True
                                                                                                        
 stack = rolliing window: accumulate and remove.
                                                                                                        
                                                                                                        
                                                                                                        
                                                                                                         
                                                                                                        
                                                                                                        
")("                                                                                                                                                                                                            
Example 2:
      Input: 
      ()
      Output: 
      true
      Explanation: 
      (). Same bracket can form balanced pairs, 
      and here only 1 type of bracket is 
      present and in balanced way.
Example 3:

Input: 
([]
Output: 
false
Explanation: 
([]. Here square bracket is balanced but 
the small bracket is not balanced and 
Hence , the output will be unbalanced.

Expected Time Complexity: O(|x|)
Expected Auixilliary Space: O(|x|)



# Python3 program to check for
# balanced brackets.

# function to check if
# brackets are balanced


def areBracketsBalanced(expr):
    stack = []

    # Traversing the Expression
    for char in expr:
        if char in ["(", "{", "["]:

            # Push the element in the stack
            stack.append(char)
        else:

            # IF current character is not opening
            # bracket, then it must be closing.
            # So stack cannot be empty at this point.
            if not stack:
                return False
 
           #### not good witring.
            current_char = stack.pop()
            if current_char == '(':
                if char != ")":
                    return False
            if current_char == '{':
                if char != "}":
                    return False





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






##############################################################################################
##############################################################################################
https://practice.geeksforgeeks.org/problems/smallest-window-in-a-string-containing-all-the-characters-of-another-string-1587115621/1/?page=1&difficulty[]=1&category[]=Strings&curated[]=1&sortBy=submissions

        Smallest window in a string containing all the characters of another string 
        Medium Accuracy: 42.59% Submissions: 48080 Points: 4
        Given two strings S and P. Find the smallest window in the string S consisting of all the characters(including duplicates) of the string P. 
        Return "-1" in case there is no such window present. In case there are multiple such windows of same length, return the one with the least starting index. 

        Example 1:

        Input:
        S = "timetopractice"
        P = "toc"
        Output: 
        toprac
        Explanation: "toprac" is the smallest
        substring in which "toc" can be found.
        Example 2:

        Input:
        S = "zoomlazapzo"
        P = "oza"
        Output: 
        apzo
        Explanation: "apzo" is the smallest 
        substring in which "oza" can be found.

 
 O(N2)
   O(N)
 
  for i1 in range(0,n) :
     for i2 in range(i1, n) : 
          to_dict(sent[i1:i2])  ==  dict_P
 
 
 #### SOlution in O(N):
 start = 0
 for i1 in range(0,n):
     
     if count =  len(P) : break
 
     if  A[i1] in set_P :  #### hash in O(1)
         count = count +1
     else :
        
         Upwork.
 
 
 
 
 


Method 1 ( Brute force solution ) 
        1- Generate all substrings of string1 ("this is a test string") 
        2- For each substring, check whether the substring contains all characters of string2 ("tist") 
        3- Finally, print the smallest substring containing all characters of string2.
  
Method 2 ( Efficient Solution ) 

        First check if the length of the string is less than the length of the given pattern, if yes then "no such window can exist ".
        Store the occurrence of characters of the given pattern in a hash_pat[].
        we will be using two pointer technique basically
        Start matching the characters of pattern with the characters of string i.e. increment count if a character matches.
        Check if (count == length of pattern ) this means a window is found.
        If such a window found, try to minimize it by removing extra characters from the beginning of the current window.
        delete one character from first and again find this deleted key at right, once found apply step 5 .
        Update min_length.
        Print the minimum length window.



# Python3 program to find the smallest window
# containing all characters of a pattern.
no_of_chars = 256

# Function to find smallest window
# containing all characters of 'pat'
def findSubString(string, pat):

    len1 = len(string)
    len2 = len(pat)

    # Check if string's length is
    # less than pattern's
    # length. If yes then no such
    # window can exist
    if len1 < len2:

        print("No such window exists")
        return ""

    hash_pat = [0] * no_of_chars
    hash_str = [0] * no_of_chars

    # Store occurrence ofs characters of pattern
    for i in range(0, len2):
        hash_pat[ord(pat[i])] += 1

    start, start_index, min_len = 0, -1, float('inf')

    # Start traversing the string
    count = 0  # count of characters
    for j in range(0, len1):

        # count occurrence of characters of string
        hash_str[ord(string[j])] += 1

        # If string's char matches with
        # pattern's char then increment count
        if (hash_str[ord(string[j])] <=     ###   freq < freq_target  : count+1
                hash_pat[ord(string[j])]):
            count += 1

        # if all the characters are matched
        if count == len2:   ###  Both match

            ### we need to move left, right.
            # Try to minimize the window
            while (hash_str[ord(string[start])] >
                   hash_pat[ord(string[start])] or
                   hash_pat[ord(string[start])] == 0):

                if (hash_str[ord(string[start])] >
                        hash_pat[ord(string[start])]):
                    hash_str[ord(string[start])] -= 1
                start += 1

            # update window size
            len_window = j - start + 1
            if min_len > len_window:

                min_len = len_window
                start_index = start

    # If no window found
    if start_index == -1:
        print("No such window exists")
        return ""

    # Return substring starting from
    # start_index and length min_len
    return string[start_index: start_index + min_len]


# Driver code
if __name__ == "__main__":

    string = "this is a test string"
    pat = "tist"

    print("Smallest window is : ")
    print(findSubString(string, pat))

# This code is contributed by Rituraj Jain





























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



















































































































































































        
    
    