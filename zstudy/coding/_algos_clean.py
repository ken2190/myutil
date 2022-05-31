



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




        Method 4 (Hashing) 
            We can solve this problem in O(m+n) time and O(m) auxiliary space. 

        
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
#############################################################################################
 16. Maximum Index 
    Medium Accuracy: 42.72% Submissions: 81979 Points: 4
    Given an array A[] of N positive integers. The task is to find the maximum of j - i subjected to the constraint of A[i] < A[j] and i < j.
    
    
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


    Your Task:
    The task is to complete the function maxIndexDiff() which finds and returns maximum index difference. 
    Printing the output will be handled by driver code. Return -1 in case no such index is found.

    Expected Time Complexity: O(N. log(n))
    Expected Auxiliary Space: O(N)

    Constraints:
    1 ≤ N ≤ 107
    0 ≤ A[i] ≤ 109
    
    
    
    hashed = 
    for i in range(n): 
    
    
    
    
    Method 3 O(nLgn):
        Use hashing and sorting to solve this problem in less than quadratic complexity after taking special care of the duplicates. 
        Approach :  
        Traverse the array and store the index of each element in a list (to handle duplicates).
        Sort the array.
        Now traverse the array and keep track of the maximum difference of i and j.
        For j consider the last index from the list of possible index of the element and for i consider the first index from the list. (As the index were appended in ascending order).
        Keep updating the max difference till the end of the array.
    
    
    
    
    # Python3 implementation of the above approach
    n = 9
    a = [34, 8, 10, 3, 2, 80, 30, 33, 1]

    # To store the index of an element.
    index = dict() 
    for i in range(n):
        if a[i] in index:
            # append to list (for duplicates)  values --List of index i
            index[a[i]].append(i)  
        else:
            # if first occurrence
            index[a[i]] = [i]   

    # sort the input array
    a.sort()       ###  A[i] < A[j]   True
    maxDiff = 0

    # Temporary variable to keep track of minimum i
    temp = n     
    for i in range(n):
        if temp > index[ a[i] ][0]:   ##### i=0   j  the max
            temp = index[a[i]][0]  
        maxDiff = max(maxDiff, index[ a[i]  ][-1]-temp)

    print(maxDiff)

    # a = [34, 8, 10, 3, 2, 80, 30, 33, 1]

    # a = [1 , 2 , 3 , 8 , 10 , 30 , 33 , 34 , 80]

    a = [6 , 3 , 1 , 5]
    sorted = [1 , 3 , 5 , 6]
    ############ 
    
    index = {
        1 :3 
        3 :2
        5: 4 
        6 :1 
    }
    
    
    a[0] =1
    index[1] = 3
    temp = min(n , 3) = 3  
    3 = index[ a[i]  ][-1]
    maxDiff = max(maxDiff, 3 - 3)
    ###  max(maxDiff, index[ a[i]  ][-1] -temp)
    
    
    a[1] =3
    index[3] = 2
    temp = min(temp=3 , 2) = 2  
    2 = index[ a[i]=3  ][-1]
    maxDiff = max(maxDiff, 2 - 2)
    
    
    
    a[2] = 5 
    index[5] = 4
    temp = min(temp=2 , 4) = 2 
    4 = index[ a[i]=5  ][-1]
    maxDiff = max(maxDiff, 4 - 2) = 2 
    
        a sorted
        a[i] <a[j]
    
        j-i  ----> only one loop

    
    ###### 
    Condition:  Largest possible value in sorted is :
    ONLY WHEN    i = 0    [0]
                j = -1   [-1]

    
    Only to check those criteria (ie duplicate)
    j-i is maximal
        
        if  index[ a[i] ][0]  < imin:   ##### i=0   j  the max
            imin = index[a[i]][0]   ### Smallest value
    
        jmax = index[ a[i]  ][-1]
        maxDiff = max(maxDiff, jmax - imin)
    
    
    ######
    How to map  (j-i)  INTO this algo  part ?
    Where it is mapped
    
    
    
    ##
    
    
    question why :  a[i]  is same in different part
    
    
    
    i <> j
    A[i] < A[j] and i < j.
    
    ###  is met  i < j.  ---> i= 0   j= -1  Last
    
    
    
    
    





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
Smallest window in a string containing all the characters of another string 
        https://practice.geeksforgeeks.org/problems/smallest-window-in-a-string-containing-all-the-characters-of-another-string-1587115621/1/?page=1&difficulty[]=1&category[]=Strings&curated[]=1&sortBy=submissions

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


    
    Expected Time Complexity: O(|S|2)
    Expected Auxiliary Space: O(1)
    

    ### Base one :
    double index,  i1, i2 --> list of substr.  check if palindrome O(N) ,  N**3.
    
    ### reduce in O(N*2)
        Double Index i1, i2
        Optmize the checking  O(1)
        s = "aaaa bbaaaa"
    
        i0 = 0 
        if n ==  1: return True
        if n == 2: 
        if  a[0] == a[0]: return True 
        else : return False   
    
        #### Min size 3 : 
        for i1 in range(1, n-1):
            start = i1 - 1
            end   = i1 + 1
    
        while  start >0 and end < n :
            if a[start] == a[i]: 
                    start -= 1 
            elif a[end] == a[i]: 
                    end += 1
    
            elif a[start] == a[end] :   ### Symmetry in middle of palindrome  a[i0-1] == ai[i0+1]
                start = start-1
                end   = end  +1
    
            else :   ### Break
                size    = end-start-1
                maxsize = max(maxsize, size)
                break

    
    ################################################### 
    ### ex:   #########################################
    'aa'  --->  case len() = 2
    'aba'  --->             
        'a'
        i = 1  		 
    odd and even
    


    ###### A O(n ^ 2) time and O(1) space program to find the# longest palindromic substring
    def longestPalSubstr(string):
        n = len(string) # calculating size of string
        if (n < 2):
            return n # if string is empty then size will be 0. # if n==1 then, answer will be 1(single

        start=0
        maxLength = 1 
        for i in range(n):
            low  = i - 1    ### i is middle one  missing the middle chracter, another 
            high = i + 1

            ### s = "aabbaa" -> b 
            ##### Need to condition on middle character.
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










 
#############################################################################################
#############################################################################################
8. Max sum in the configuration 
            Medium Accuracy: 49.48% Submissions: 51212 Points: 4
            Given an array(0-based indexing), you have to find the max sum of i*A[i] where A[i] is the element at index i in the array. 
            The only operation allowed is to rotate(clock-wise or counter clock-wise) the array any number of times.

        Example 1:
            Input:
            N = 4
            A[] = {8,3,1,2}
            Output: 29
            Explanation: Above the configuration
            possible by rotating elements are
            3 1 2 8 here sum is 3*0+1*1+2*2+8*3 = 29
            1 2 8 3 here sum is 1*0+2*1+8*2+3*3 = 27
            2 8 3 1 here sum is 2*0+8*1+3*2+1*3 = 17
            8 3 1 2 here sum is 8*0+3*1+1*2+2*3 = 11
            Here the max sum is 29 
            Your Task:
            Your task is to complete the function max_sum which takes two arguments which is the array A [ ] 
            and its size and returns an integer value denoting the required max sum.

        Expected Time Complexity: O(N).
        Expected Auxiliary Space: O(1).

        Constraints:
        1<=N<=104
        1<=A[]<1000
        

        
        
        
        Method 3: The method discusses the solution using pivot in O(n) time. 
        The pivot method can only be used in the case of a sorted or a rotated sorted array. 
        For example: {1, 2, 3, 4} or {2, 3, 4, 1}, {3, 4, 1, 2} etc.


        Approach: Let's assume the case of a sorted array. 
        As we know for an array the maximum sum will be when the array is sorted in ascending order. 
        In case of a sorted rotated array, we can rotate the array to make it in ascending order. 
        So, in this case, the pivot element is needed to be found following which the maximum sum can be calculated.

        Algorithm: 
        Find the pivot of the array: if arr[i] > arr[(i+1)%n] then it is the pivot element. 
        (i+1)%n is used to check for the last and first element.
        After getting pivot the sum can be calculated by finding the difference with the pivot which will be the multiplier 
        and multiply it with the current element while calculating the sum
        Implementations:
        
        
        # Python3 program to find maximum sum of  all rotation of i*arr[i] using pivot. 

        # function definition 
        def maxSum(arr, n) :
            sum = 0
            pivot = findPivot(arr, n)

            # difference in pivot and index 
            # of last element of array 
            diff = n - 1 - pivot 
            for i in range(n) :
                sum1 = sum1 + ((i + diff) % n) * arr[i]; 
            return sum1
            
        
        # function to find pivot 
        def findPivot(arr, n) :
            for i in range(n) : 
                if(arr[i] > arr[(i + 1) % n]) :
                    return i; 

        
        
        # Driver code 
        if __name__ == "__main__" :

            # rotated input array 
            arr = [8, 3, 1, 2] 
            n= len(arr) 
            
            max= maxSum(arr, n)
            print(max)

        # This code is contributed by Ryuga



















##############################################################################################
##############################################################################################
 partition the string into as many parts as possible so that each letter appears in at most one part.
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

            
            
            
            
    
    

 
 
####################################################################################
####################################################################################
 21. Longest Substring Without Repeating Characters 
        Medium Accuracy: 45.75% Submissions: 12262 Points: 4
        Given a string S, find the length of its longest substring that does not have any repeating characters.

    Example 1:
        Input:
        S = gee ksforgeeks
        Output: 7
        Explanation: The longest substring
        without repeated characters is "ksforge".
    
    
    Example 2:
        Input:
        S = abbcdb
        Output: 3
        Explanation: The longest substring is
        "bcd". Here "abcd" is not a substring
        of the given string.
        Your Task:
        Complete SubsequenceLength function that takes string s as input and returns the length of the longest substring that does not have any repeating characters.

    Expected Time Complexity: O(N)
    Expected Auxiliary Space: O(1)

    Constraints: 
    0<= N <= 10^5
    here, N = S.length
    

    2 pointers  
        for i in range(0,n)  :  O(N)
        2 pointer i1,i2  for substring
        condition :  c[i]  not in ddict
        keep the state.
    
    max1 =0 
    ddict = {}
    st = 0
    for i in range(0,n):
    
        ci = S[i]
    
        if ci not in ddict :    ### condition1 :  Unique character
            ddict[ci] = 1
            maxl = max(max1 , (i - st + 1))
            # continue  ### increase lenght
        else :
        ddict[ S[st] ] -= 1   ###  Decrement
        st   = st + 1
            i   = i - 1          ### why ? because next iteration increase the counter., need to keep the counter same.
    
    return max1
    

    #########
    
    
    
    
    Python3 program to find the length
    # of the longest substring
    # without repeating characters
    def longestUniqueSubsttr(string):

        # last index of every character
        last_idx = {}   ###
        max_len = 0
    
        # starting index of current
        # window to calculate max_len
        start_idx = 0
    
        for i in range(0, len(string)):
        
            # Find the last index of str[i]
            # Update start_idx (starting index of current window)
            # as maximum of current value of start_idx and last
            # index plus 1
            if string[i] in last_idx:   ### Why ????  :  your dog making noise, cannot hear you....
                ##  last_idx is dictionary 
                # the solution is simply looking for the last index of a certain character. 
                    Then t
                #### did not see it was fict,.,,,  very BAD naming... last_idc ==  index !!!
                start_idx = max(start_idx, last_idx[string[i]] + 1)
    
            # Update result if we get a larger window
            max_len = max(max_len, i-start_idx + 1)
    
            # Update last index of current char.
            last_idx[string[i]] = i
    
        return max_len
    
    
    # Driver program to test the above function
    string = "geeksforgeeks"
    print("The input string is " + string)
    length = longestUniqueSubsttr(string)
    print("The length of the longest non-repeating character" +
        " substring is " + str(length))
    
    
    https://www.geeksforgeeks.org/length-of-the-longest-substring-without-repeating-characters/
    
    
    
        







    
    
    
    
##############################################################################################
##############################################################################################    
A string s can be partitioned into groups of size k using the following procedure:
        The first group consists of the first k characters of the string, the second group

        consists of the next k characters of the string, and so on. Each character can be a part of exactly one group.
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

            









###################################################################################################################
###################################################################################################################
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









    
    
##############################################################################################
##############################################################################################
2104. Sum of Subarray Ranges
    https://leetcode.com/problems/sum-of-subarray-ranges/
    https://leetcode.com/problems/sum-of-subarray-ranges/discuss/1624416/Python3-stack
    
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
        
    
        
        
    #######  O(N**2)   
    class Solution:
        def subArrayRanges(self, nums: List[int]) -> int:
            
            def fn(op): 
                """Return min sum (if given gt) or max sum (if given lt)."""
                ans = 0 
                stack = []
                for i in range(len(nums) + 1): 
                    while stack and (i == len(nums) or op(nums[stack[-1]], nums[i])): 
                        mid = stack.pop()
                        ii = stack[-1] if stack else -1 
                        ans += nums[mid] * (i - mid) * (mid - ii)
                    stack.append(i)
                return ans 
            
            return fn(lt) - fn(gt)    
        

    For those not understanding multiplication : i-mid * mid-i . Forget minima/maxima for a second and focus on why it's multiplied.

    We want to see the number of subarray's nums[mid] would be a part of.

    For the left side, it will be a part of subarray's with mid as the ending.
    For the right side, it will be a part of subarrays with mid as the beginning.
    Now total subarray's that can be formed is by the combination formula, for each subarray in left side, how many subarrays on the right side can it be combined with.
    i.e,
    each subarray on the left can pick every subarray on the right , leading to left*ri   
        
        
        
        
        



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









##############################################################################################
##############################################################################################
Permutation is an arrangement of objects in a specific order. 

        def permutation(lst):
            if len(lst) == 0:
                return []
        
            
            if len(lst) == 1:  # If there is only one element in lst then, onlone permutation is possible
                return [lst]

        
            l = [] 
            for i in range(len(lst)):
            m = lst[i]
        
            # Extract lst[i] or m from the list.  remLst is remaining list
            remLst = lst[:i] + lst[i+1:]
        
            # Generating all permutations where m is first element
            for p in permutation(remLst):
                l.append([m] + p)
            return l
        

        ### Main
        data = list('123')
        for p in permutation(data):
            print (p)
            





##############################################################################################
##############################################################################################
Palindrome
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
            
            
    







#################################################################################### 
####################################################################################
 20. Container With Most Water 
    https://practice.geeksforgeeks.org/problems/container-with-most-water0535/1/
    
          Medium Accuracy: 53.18% Submissions: 10804 Points: 4
          Given N non-negative integers a1,a2,....an where each represents a point at coordinate (i, ai). 
          N vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i,0).
          Find two lines, which together with x-axis forms a container, such that it contains the most water.

    Example 1:
          Input:
          N = 4
          a[] = {1,5,4,3}
          Output: 6
          Explanation: 5 and 3 are distance 2 apart.
          So the size of the base = 2. Height of
          container = min(5, 3) = 3. So total area
          = 3 * 2 = 6.

    Example 2:
          Input:
          N = 5
          a[] = {3,1,2,4,5}
          Output: 12
          Explanation: 5 and 3 are distance 4 apart.
          So the size of the base = 4. Height of
          container = min(5, 3) = 3. So total area
          = 4 * 3 = 12.
    Your Task :
    You only need to implement the given function maxArea. Do not read input, instead use the arguments given in the function and return the desired output. 

    Expected Time Complexity: O(N).
    Expected Auxiliary Space: O(1).

    Constraints:
    2<=N<=105
    1<=A[i]<=100
    
    
    
      Condition :   Max(  (i2-i1)* min(h1, h2)   )  at each step.
        
        H  : ehight array
        i2 = n - 1
        for i1 in range(0, i2):
    
          curr_volume =  (i2-i1)*min(H[i1], H[i2]) 
    
          if curr_volume < max1 :
              i2 = i2 - 1 #  most right position of array, so we decrement it if the condition was not met. and start over again with a different v  
              
              i1 = i1 - 1  ### next step is i1+1, so keeep same value
              continue
      
          else :
            max1 = Max( max1,   (i2-i1)*min(H[i1], H[i2]) )
    
    
    condition1 :
      current_volume =  (i2-i1)*min(H[i1], H[i2]) 
    
    
    

    Time Complexity: O(n). 

    
    # Python3 code for Max 
    # Water Container
    def maxArea( A):
        l = 0
        r = len(A) -1
        area = 0
        
        while l < r:
            # Calculating the max area
            area = max(area, min(A[l], 
                            A[r]) * (r - l))
        
            if A[l] < A[r]:  #### condtion ????  :why 
              ### why not checking the volume.....
              ### equivalent to volume. :  
              ###   (i2-i1)*min(H[i1], H[i2])  < max1   Not the same conditionss.  
              ###  ok, thats, good, it was to  we dont miss anything 
              l += 1
            else:
                r -= 1
        return area

    
    
    
    # Driver code
    a = [1, 5, 4, 3]
    b = [3, 1, 2, 4, 5]

    print(maxArea(a))
    print(maxArea(b))


    





 
 
#################################################################################### 
####################################################################################
Smallest distinct window 
    Medium Accuracy: 50.29% Submissions: 21827 Points: 4
    Given a string 's'. The task is to find the smallest window length that contains all the characters of the given string at least one time.
    For eg. A = aabcbcdbca, then the result would be 4 as of the smallest window will be dbca.

    Example 1:

          Input : "AABBBCBBAC"
          Output : 3
          Explanation : Sub-string -> "BAC"
    
    
    Example 2:
          Input : "aaab"
          Output : 2
          Explanation : Sub-string -> "ab"

    
    Example 3:
          Input : "GEEKSGEEKSFOR"
          Output : 8
          Explanation : Sub-string -> "GEEKSFOR"



    Your Task:  
    You don't need to read input or print anything. Your task is to complete the function findSubString() 
    which takes the string  S as input and returns the length of the smallest such window of the string.


    Expected Time Complexity: O(256.N)
    Expected Auxiliary Space: O(256)

    ddict [0...256] character
    smallest:
    
        
      charctes_in_string = {} 
      for c in S: 
            charctes_in_string[c] = 1 
    
      i2 = 0
      for i2 in range(0, n):
      
        start from smallest--> increase
        condition all characters:

        ####ci  any idea on checkin all the chracers      
        array[ci] += 1 
    
        #Condition 
        all_chars_found = True 
        for char in charctes_in_string: 
            if not array[char]: 
                all_chars_found = False
                break 
    
        if all_chars_found:  
            
            while condition:   ### condition, checking if the substring between i1 to i2 has all the charactes of the original string at least one
                i1 += 1 
                array[ci] -= 1 
            min1 =  min(min1,   i2-i1+1)
            ###  In the middle or right , can have another small window, 
            #### So, need to continue. 
        else : 
            ### nothing to do, increment i2 to find more characters to satisfy the condition. 


    ### Open this one:
    https://github.com/qiyuangong/leetcode
    
    
    https://leetcode.com/problems/reverse-words-in-a-string-ii/
    https://github.com/qiyuangong/leetcode/blob/master/python/186_Reverse_Words_in_a_String_II.py
    
    
    For Next time
    
    
    
    https://leetcode.com/problems/largest-divisible-subset/
    
    
    
    
    S = "SSSSBAC" smalles string = "SBAC"
    SSSSBAC
    i1 = 0 , i2 = 6 
    
    
    
    
    
    https://www.geeksforgeeks.org/smallest-window-contains-characters-string/
    Python program to find the smallest
    # window containing# all characters of a pattern
    from collections import defaultdict
    
    MAX_CHARS = 256
      
    def findSubString(strr):
    
        n = len(strr)
    
        # if string is empty or having one char
        if n <= 1:
            return strr
    
        # Count all distinct characters.
        dist_count = len(set([x for x in strr]))
    
        curr_count = defaultdict(lambda: 0)
        count = 0
        start = 0
        min_len = n
    
        # Now follow the algorithm discussed in below
        # post. We basically maintain a window of characters
        # that contains all characters of given string.
        for j in range(n):
            curr_count[strr[j]] += 1
    
            # If any distinct character matched,
            # then increment count
            if curr_count[strr[j]] == 1:
                count += 1
    
            # Try to minimize the window i.e., check if
            # any character is occurring more no. of times
            # than its occurrence in pattern, if yes
            # then remove it from starting and also remove
            # the useless characters.
            if count == dist_count:
                while curr_count[strr[start]] > 1:
                    if curr_count[strr[start]] > 1:
                        curr_count[strr[start]] -= 1
    
                    start += 1
    
                # Update window size
                len_window = j - start + 1
    
                if min_len > len_window:
                    min_len = len_window
                    start_index = start
    
        # Return substring starting from start_index
        # and length min_len """
        return str(strr[start_index: start_index +
                        min_len])
    
    










  
 
 
 ######################################################################################
 ######################################################################################A string s can be partitioned into groups of size k using the following procedure.

      The first group consists of the first k characters of the string, the second group consists of the next k characters of the string, and so on. 
      Each character can be a part of exactly one group.
      For the last group, if the string does not have k characters remaining, 
      a character fill is used to complete the group.
      Note that the partition is done so that after removing the fill character from the last group (if it exists) 
      and concatenating all the groups in order, the resultant string should be s.

      Given the string s, the size of each group k and the character fill, 
      return a string array denoting the composition of every group s has been divided into, using the above procedure.
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

          







    

 
 
 
######################################################################################
###################################################################################### 
18. Kadane's Algorithm 
    Medium Accuracy: 51.0% Submissions: 100k+ Points: 4
    Given an array Arr[] of N integers. Find the contiguous sub-array(containing at least one number) which has the maximum sum and return its sum.
    https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1/?

    Example 1:
          Input:
          N = 5
          Arr[] = {1,2,3,-2,5}
          Output:
          9
          Explanation:
          Max subarray sum is 9
          of elements (1, 2, 3, -2, 5) which 
          is a contiguous subarray.

    
    Example 2:
        Input:
        N = 4
        Arr[] = {-1,-2,-3,-4}
        Output:
        -1
        Explanation:
        Max subarray sum is -1 
        of element (-1)

    Your Task:
    Expected Time Complexity: O(N)
    Expected Auxiliary Space: O(1)


    Constraints:
    1 ≤ N ≤ 106
    -107 ≤ A[i] ≤ 107
    
    
    
    Kadane's Algorithm:
    Initialize:

        max_so_far = INT_MIN

        max_ending_here = 0



    Loop for each element of the array
      (a) max_ending_here = max_ending_here + a[i]
      (b) if(max_so_far < max_ending_here)
                max_so_far = max_ending_here

      (c) if(max_ending_here < 0)
                max_ending_here = 0

    return max_so_far
    

    
    # Python program to find maximum contiguous subarray
    
    # Function to find the maximum contiguous subarray
    from sys import maxint
    def maxSubArraySum(a,size):
        
        max_so_far = -maxint - 1
        max_ending_here = 0
        
        for i in range(0, size):
            max_ending_here = max_ending_here + a[i]
            if (max_so_far < max_ending_here):
                max_so_far = max_ending_here

            if max_ending_here < 0:
                max_ending_here = 0   
        return max_so_far
    
    # Driver function to check the above function 

    a = [-2, -3, 4, -1, -2, 1, 5, -3]

    print "Maximum contiguous sum is", maxSubArraySum(a,len(a))
    
    #This code is contributed by _Devesh Agrawal_

    
    
    
 
 
 



#####################################################################################  
#####################################################################################
Given a signed 32-bit integer x, return x with its digits reversed. 
    If reversing x causes the value to go outside the signed 32-bit integer range [-2**31, 2**31 - 1], 

    then return 0.

    Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

    
    Example 1:

        Input: x = 123
        Output: 321

        Input: x = -123
        Output: -321

        Input: x = 120
        Output: 21
    

    Constraints:
      -2**31 <= x <= 2**31 - 1
      
      
      
      def solution(number: int): 
        
        number_str = str(number)
        
        offset = (number_str[0] == '-') 
        
        return if offset '-' + number_str[::-1] else number_str[::-1]
        
      
      
      
      class Solution:
        def reverse(self, x: int) -> int:
            return self.reverseIntegerByList(x)
        
        # O(n) || O(1) 34ms 86.01%
        def reverseIntegerByList(self, value):
            if value == 0:return value
            isNeg = value < 0
            value = abs(value)
            numList = list()
            newReverseNumber = 0
            while value > 0:
                k = value % 10  #### %10 to get the digitis.
                newReverseNumber = newReverseNumber * 10 + k
                value //= 10

            if newReverseNumber >= 2**31 or newReverseNumber >= 2**31 - 1:
                return 0

            return newReverseNumber if not isNeg else -newReverseNumber
        

        # Brute force
        # O(n) || O(m) 37ms 76.28%
        #  m stands for making it string
        def reverseIntegerByString(self, value):
            isNeg = value < 0
            strVal = str(abs(value))
            strVal = int(strVal[::-1])
            if strVal >= 2**31 or strVal >= 2**31 - 1:
                return 0
            return -strVal if isNeg else strVal
      










#######################################################################################################  
#######################################################################################################
1249. Minimum Remove to Make Valid Parentheses

    HOMEWORK 

        https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/solution/  
        Medium

        Given a string s of '(' , ')' and lowercase English characters.

        Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid 
        and return any valid string.

        Formally, a parentheses string is valid if and only if:

        It is the empty string, contains only lowercase characters, or
        It can be written as AB (A concatenated with B), where A and B are valid strings, or
        It can be written as (A), where A is a valid string.
  

  Example 1:
      Input: s = "lee(t(c)o)de)"
      Output: "lee(t(c)o)de"
      Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.

  Example 2:
      Input: s = "a)b(c)d"
      Output: "ab(c)d"
      Example 3:

      Input: s = "))(("
      Output: ""
      Explanation: An empty string is also valid.
  

  Constraints:
        1 <= s.length <= 10**5
        s[i] is either'(' , ')', or lowercase English letter.
    
    
    
    
    
    def solution(word: str) -> int: 
      
      open1 = 0 , closes = 0 
      to_remove = 0 
      valid_string = "" 
      for ch in word: 
        
        if ch == '(': 
          #TODO 
          valid_string += '('
          open1 += 1 
        else if ch == ')': 
          #TODO 
          if open1 > 0: 
            open1 -= 1  ## Close the brracket, reser state. 
            valid_string += ')'
          else: 
            to_remove += 1    ### no open, only )
        else: 
          valid_string += ch 
      
      remove_opens = open1 
      while remove_opens: 
        #remove the last opening brackets with number remove opens  TODO
        remove_opens -=  1 
      to_remove += open1
      
      return to_remove 
  


  Test Cases: 
    ))) ((( 
    

    
    ##### Solution 2
    Approach 2: Two Pass String Builder
  Intuition

  A key observation you might have made from the previous algorithm is that for all invalid ")", 
      we know immediately that they are invalid (they are the ones we were putting in the set).
      It is the "(" that we don't know about until the end (as they are what was left on the stack at the end). W
      e could be building up a string builder in that first loop that has all of the invalid ")" removed. This would be half the problem solved in the first loop, in O(n)O(n) time.
    
  class Solution:
      def minRemoveToMakeValid(self, s: str) -> str:

          def delete_invalid_closing(string, open_symbol, close_symbol):
              sb = []
              balance = 0
              for c in string:
                  if c == open_symbol:
                      balance += 1
      
                  if c == close_symbol:
                      if balance == 0:
                          continue
                      balance -= 1
                  sb.append(c)
              return "".join(sb)

          # Note that s[::-1] gets the reverse of s.
          s = delete_invalid_closing(s, "(", ")")
          s = delete_invalid_closing(s[::-1], ")", "(")
          return s[::-1]
    
    








########################################################################################################  
########################################################################################################
92. Top K Frequent Words
   Medium
    Given an array of strings words and an integer k, return the k most frequent strings.
    Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.

 
    Example 1:
        Input: words = ["i","love","leetcode","i","love","coding"], k = 2
        Output: ["i","love"]
        Explanation: "i" and "love" are the two most frequent words.
        Note that "i" comes before "love" due to a lower alphabetical order.

        
    Example 2:
        Input: words = ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4
        Output: ["the","is","sunny","day"]
        Explanation: "the", "is", "sunny" and "day" are the four most frequent words, with the number of occurrence being 4, 3, 2 and 1 respectively.


    Constraints:
        1 <= words.length <= 500
        1 <= words[i] <= 10
        words[i] consists of lowercase English letters.
        k is in the range [1, The number of unique words[i]]
    

    Follow-up: Could you solve it in O(n log(k)) time and O(n) extra space?
    
    
    Dictoiwht frequncy
    
    






  
########################################################################################################  
########################################################################################################
39. Combination Sum
            Medium
            https://leetcode.com/problems/combination-sum/solution/


            Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates
            where the chosen numbers sum to target. You may return the combinations in any order.
            The same number may be chosen from candidates an unlimited number of times. 
            Two combinations are unique if the frequency of at least one of the chosen numbers is different.
            It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.



        Example 1:
            Input: candidates = [2,3,6,7], target = 7
            Output: [[2,2,3],[7]]
            
            Explanation:
            2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
            7 is a candidate, and 7 = 7.
            These are the only two combinations.

            
        Example 2:
            Input: candidates = [2,3,5], target = 8
            Output: [[2,2,2,2],[2,3,3],[3,5]]
            Example 3:

            Input: candidates = [2], target = 1
            Output: []
        

        Constraints:
            1 <= candidates.length <= 30
            1 <= candidates[i] <= 200
            All elements of candidates are distinct.
            1 <= target <= 500

        
        
        #### Condiitons :
            a[i]> T  : remove
            sort higheest number :
            
            targetnew = target-a[i]
            
            Conidiotn 0
                target in ddict :  return target (in list) 
            
            pick one valid nmber in ddict:  a[k]
                
                res  = algo(ddict,  targetnew = target-a[k])
            
                Merge( a[k]  )
            
            zoom is cut, let resend here
            

            
        
        Solution
        Overview
        This is one of the problems in the series of combination sum. They all can be solved with the same algorithm, i.e. backtracking.
        Before tackling this problem, we would   
        Algorithm

            As one can see, the above backtracking algorithm is unfolded as a DFS (Depth-First Search) tree traversal, which is often implemented with recursion.
            Here we define a recursive function of backtrack(remain, comb, start) (in Python), which populates the combinations, starting from the current combination (comb), the remaining sum to fulfill (remain) and the current cursor (start) to the list of candidates. Note that, the signature of the recursive function is slightly different in Java. But the idea remains the same.
            
        O(2**N): recursive at O(N)  for each loop 
            

        https://leetcode.com/problems/combination-sum/submissions/
            
        ##### DFS  : passed OK
        class Solution:
        def combinationSum(self, candidates, target):
            """
            :type candidates: List[int]
            :type target: int
            :rtype: List[List[int]]
            """
            res = []
            candidates.sort()

            def dfs(target, index, path):
                if target < 0:
                    return  # backtracking
                if target == 0:
                    res.append(path)
                    return 
                for i in range(index, len(candidates)):
                    dfs(target-candidates[i], i, path+[candidates[i]]  )

            dfs(target, 0, [])
            return res
            
            
            
        #### DP Version : passed OK
        class Solution:
            def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

                ### DP for all the sub-target [ 0,  target]
                dp = [[] for _ in range(target + 1)]
                
                for index in range(len(candidates)):    
                    for sub_target in range(1, target + 1):  ## start from Smallest
                
            
                        if candidates[index] == sub_target:
                            dp[sub_target].append([candidates[index]])
                            
                        elif candidates[index] < sub_target:
                            remain_sub_target = sub_target - candidates[index]

                            for remain_possibility in dp[remain_sub_target]:  ### Already compute before of the DP
                                dp[sub_target].append(  [candidates[index]] + remain_possibility )   ### sub_target =  current_one + previous_target
                
                return dp[target]
            
            
        
        Similar problem
        Subsets
        Subsets II
        Permutations
        Permutations II
        Combinations
        Combination Sum II
        Combination Sum III
        Palindrome Partition


            
    
    
 




#########################################################################################    
#########################################################################################
1048. Longest String Chain
    https://leetcode.com/problems/longest-string-chain/discuss/1213876/Python-3-solutions-LIS-DP-Top-down-DP-Bottom-up-DP-Clean-and-Concise
    https://leetcode.com/problems/longest-string-chain/
        
    You are given an array of words where each word consists of lowercase English letters.

    wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA
        without changing the order of the other characters to make it equal to wordB.

    For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".
    A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.

    Return the length of the longest possible word chain with words chosen from the given list of words.

    

    Example 1:
          Input: words = ["a","b","ba","bca","bda","bdca"]
          Output: 4
          Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
        
        
    Example 2:
          Input: words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
          Output: 5
          Explanation: All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].

        
    Example 3:
        Input: words = ["abcd","dbqca"]
        Output: 1
        Explanation: The trivial word chain ["abcd"] is one of the longest word chains.
        ["abcd","dbqca"] is not a valid word chain because the ordering of the letters is changed.
    

    Constraints:
        1 <= words.length <= 1000
        1 <= words[i].length <= 16
        words[i] only consists of lowercase English letters.    
        
        

        cond :  word1 in word2 
                pick up word2, word2 sequentially.
        
        1) sort the list lword : Log(n. logn) 
        
        word1 = lword[0]
        sep.append(word1)

        #### Not correct:    
        for i in range(0,N) :
        
            if seq= [] :
                seq.append(lword[i])
        
            if is_pred(seq[-1], lword[i]):
                  seq.append(lword[i])    
            else :
                #### case  1: seq[-1] is bad ???
        
        
                #### case 2:   lword[i] is bad:
                    ## do nothing
        
        
          ### Need recursion in O(N2)
        
          
    ✔️ Solution 2: Top-Down DP
    Let dp(word) be the length of the longest possible word chain end at word word.
    To calculate dp(word), we try all predecessors of word word and get the maximum length among them.


    class Solution:
        def longestStrChain(self, words: List[str]) -> int:
            wordSet = set(words)

            @lru_cache(None)
            def dp(word):
                ans = 1
                for i in range(len(word)):
        
                    ### trick here, 
                    ### Equivalent to text dscripition
                    predecessor = word[:i] + word[i + 1:]  ### remove middle letter
        
                    if predecessor in wordSet:
                        ans = max(ans, dp(predecessor) + 1)
                return ans

            return max(dp(w) for w in words)   
      
    ####here   
        WordSet: is the words given in the input (OK?). 
        predecessor = word[:i] + word[i + 1:]  ### remove middle letter
        predecerro in wordSet
        
        Example: 
          "abc" is a predecessor of "abac"
          "abac" -> ".bac" and checking if it's inside Wordset 
        
        Take target word "abac":
          we list all possible  (n-1) predecssor of 'abac'
            --> Equivalent to :
              remove 1 char ad check if this is in our wordset.
            Nb of (N-1) chaacter predicto is 
              Ni = count()
                  ans = max(ans, dp(predecessor) + 1)            
        
            Calculate the same for each predc  (ie N-2 pred of  predecior)
        
        
            ### formulae:  
            NPredecessor = Max(  Count(N-1 char Predecssor) for word in wordlist) )
                            
                          return max(dp(w) for w in words)   
        
        
                        
        Part 1)
          we need to map  Condition of the text descrptions --> Code Loop , Code Block
                #### we can parse all predecssor 
                ####        predecessor = word[:i] + word[i + 1:]  ### remove middle letter


        Part 2) 
          We need to map  the "loops in the code",    INTO  complexity O( ....) 
              "Piece of Code" --->  Complexity O(....)

          ################## 
          O(N) : N words  ###  max(dp(w) for w in words)   
              DP is called at most (N*L) times  :    ####  for character -->  if predecessor in wordSet: is (O(N))
              ### implicift complexity 
              
            total=      O(N)  * O(N*L)


        "wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA
        without changing the order of the other characters to make it equal to wordB.""

            
        ### Equivalent to text dscripition
        predecessor = word[:i] + word[i + 1:]  ### remove middle letter
        
        
        
        
        ok its ok,I can see your typinf
    Complexity
        Time: O(N^2 * L), where N <= 1000 is number of words, L <= 16 is length of each word.
        Space: O(N)    

          
        
        
        
    ✔️ Solution 1: Longest Increasing Subsequence Idea
    Time: O(N^2 * L), where N <= 1000 is number of words, L <= 16 is length of each word.
    Space: O(N)

    Firstly, we sort words in increasing order by their length, to ensure length of the previous word no longer
    than length of the current word, it means len(words[j]) <= len(word[i]), where j < i.
    Let dp[i] be the longest increasing subsequence which ends at words[i].
    To check if word1 is a predecessor of word2, we need to check
    len(word1) + 1 must equal to len(word2).
    word1 must be a subsequence of word2.

    class Solution:
        def longestStrChain(self, words: List[str]) -> int:

            def isPredecessor(word1, word2):
                if len(word1) + 1 != len(word2): return False
                i = 0
                for c in word2:
                    if i == len(word1): return True
                    if word1[i] == c:
                        i += 1
                return i == len(word1)
            
            words.sort(key=len)
            n = len(words)
            dp = [1] * n
            ans = 1
            
            for i in range(1, n):   ## All words
                for j in range(i):   ### All words up to i
                    if isPredecessor(words[j], words[i]) 
                    and dp[i] < dp[j] + 1:   ### why ???  
                        dp[i] = dp[j] + 1
                ans = max(ans, dp[i])
            return ans
    Complexity

        ### and dp[i] < dp[j] + 1:   ### why ???
        dp[i] is the longest chain ending in position i 
        if longer chain is found 
        
        
        
        
      







 
#################################################################################### 
####################################################################################
23. Max Circular Subarray Sum 
      https://leetcode.com/problems/maximum-sum-circular-subarray/discuss/1348545/Python-3-solutions-Clean-and-Concise-O(1)-Space
      Hard Accuracy: 45.16% Submissions: 48463 Points: 8
      Given an array arr[] of N integers arranged in a circular fashion. 
      Your task is to find the maximum contiguous subarray sum.

                            
    Example 1:
          Input:
          N = 7
          arr[] = {8,-8,9,-9,10,-11,12}
          Output:
          22
          Explanation:
          Starting from the last element
          of the array, i.e, 12, and 
          moving in a circular fashion, we 
          have max subarray as 12, 8, -8, 9, 
          -9, 10, which gives maximum sum 
          as 22.

          
    Example 2:
          Input:
          N = 8
          arr[] = {10,-3,-4,7,6,5,-4,-1}
          Output:
          23
          Explanation: Sum of the circular 
          subarray with maximum sum is 23

          
    Your Task:
          The task is to complete the function circularSubarraySum() which returns a sum of the circular subarray with maximum sum.

          Expected Time Complexity: O(N).
          Expected Auxiliary Space: O(1).


    Constraints:
        1 <= N <= 106
        -106 <= Arr[i] <= 106
    
      
      Array parsing
        
        
        A[N-1  % N]
        A[ N  %N = 0 ]  = A[0]
        A[N+1 %N]  = A[1]


      standard Array :
          
          for i2 in range(0, N):
              
              while cur_total >= max_total and i1 < i1 :
                i1 = i1 + 1              
                cur_total = sum(A[i1:i2])                
                max_total = max(max_total, cur_total) 
                  
    neg. value ???
    minimu continous array 
      largest minimum value

    If only positive ---> whole array
    Indetify the largest negative array Continous.


    [5, 5, 5 ,5, 5] => The whole array 
    [5, 5, ,5, -5, 5]
                  
      [ 4,1, 4, -10, 1, -5, 20, -1 ]  --> 20,-1,4,1,4  
        WHY you take this approach , benfits of this approach.
      
      
  
        
    # Python program for maximum contiguous circular sum problem Standard Kadane's algorithm 
    ##  to find maximum subarray sum
    def kadane(a):
        Max = a[0]
        temp = Max
        for i in range(1,len(a)):
            temp += a[i]
            if temp < a[i]:
                temp = a[i]
            Max = max(Max,temp)
        return Max

      
    1) why we cnnot kadane direcrtly on max sum ??
        
        Because the problem is circular array, 
            you can make the % trick instead 
      
      
      2) why we can apply Kadane on the maximum negative ?
          still circular here too....Only no-circular array
        
          Why you dont need to this the % trick....
          
          Because he is finding both the min and max of the array (-ve and +ve) 
          and getting the solution from subtracting them. 
          
          use Kadane to find both min and max ???
            
            
          
      
    # The function returns maximum circular contiguous sum in a[]
    def maxCircularSum(a):

        n = len(a)

        # Case 1: get the maximum sum using standard kadane's
        # algorithm
        max_kadane = kadane(a)   #####  Max array

        # Case 2: Now find the maximum sum that includes elements.
        
        # You can do so by finding the maximum negative contiguoussum
        # convert a to -ve 'a' and run kadane's algo
        neg_a = [-1*x for x in a]
        max_neg_kadane = kadane(neg_a)   ### max negative. 

        
        # Max sum with corner elements will be:
        # array-sum - (-max subarray sum of inverted array)
        max_wrap = -(sum(neg_a)-max_neg_kadane)  
        ### sum_all_array - max_negative_subbrary_sum 
        ###  circular : max_wrap is related to the circula condition.
        
        
        

        # The maximum circular sum will be maximum of two sums
        res = max(max_wrap,max_kadane)
        return res if res != 0  else max_kadane

    # Driver function to test above function
    a = [11, 10, -20, 5, -3, -5, 8, -13, 10]
    print "Maximum circular sum is", maxCircularSum(a)

      [ 4,1, 4, -10, 1, -5, 20, -1 ]  --> 20,-1,4,1,4  

      [ -5, 20, -1,  4,1, 4, -10, 1, ]  --> 
      
            max negativ :  -10, 1,-5
            max kadane  :  20,-1,4,1,4    : never circular, why ?  Because Kadane apply says to [0, N], only straight array

                
            2 possibilities :
                1)  max array is NOT circular -->  max array  = Kadane
              
                2)  max arrat IS circular  --. Max array NOT kadane
                            BUT, Max negative is NOT circular :  (cannot have both circular )
                                  --> we apply kadane on Max Negative
                                          Max array = total array - Max Negative.
                        
                  3) merge both cases
                                Max(case1, case2, ....)
                  
                  We do not need TO know if max array is circular or NOT.
                    to solve the problem (ie implicit).
                    
                    
                  Should take few example (instead of writing )

                  

                        Why we can apply kadane
                        why we need to compute the negative max array.
                          
                    










#################################################################################################
#################################################################################################
https://leetcode.com/problems/kth-largest-element-in-an-array/
    Given an integer array nums and an integer k, return the kth largest element in the array.
        Note that it is the kth largest element in the sorted order, not the kth distinct element.
        https://leetcode.com/problems/kth-largest-element-in-an-array/
    Example 1:
        Input: nums = [3,2,1,5,6,4], k = 2
        Output: 5
        Example 2:

    Input:
        nums = [3,2,3,1,2,4,5,5,6], k = 4
        Output: 4
        
        
      O(N.log N)  : sort and take top-k
          
      O(N)    
        
        median:    K = N/2    
        median in un-sorted array,  K= N/2 odd and even in O(N)
        
        
        
    class Solution:
        def findKthLargest(self, nums, k):
            if not nums: return
            pivot = random.choice(nums)
            left =  [x for x in nums if x > pivot]   ###   x > pivot
            mid  =  [x for x in nums if x == pivot]
            right = [x for x in nums if x < pivot]    ## right < pivotr
            
            L, M = len(left), len(mid)
            
            if k <= L:   #### solution is on the left 
                return self.findKthLargest(left, k)
              
            elif k > L + M:  ## must be on the right
                return self.findKthLargest(right, k - L - M)  newk = remove the (l+M) elements before element k,  due size of right
            else:
                return mid[0]
        
        O(N)  =   (N + N/2 + N/4 + N/8 + .... )   sum( 1/2**k)  < 1  =   1-1/2 / (1 - 1/2**N)
        
      Complexity: time complexity is O(n) in average, because on each time we reduce
      searching range approximately 2 times. This is not strict proof,
          for more details you can do some googling. Space complexity is O(n) as well.  
        
        

        MergeSort
        
          How the merge is done ?
          
            if a[0] > a[1]  : swap(a[0], a[1])

              
        repeat recursively from subarray of size = 2 , 4 , 8 , .... , until you reach N 
          Sort(left),  sort(right)
          merge(left_sorted, right_sorted)  : can you clarify this one
              
              left and right are already sorted -> starting each subarray (left) check who is larger and set left[0] and right[0] 
              Merge is O(N)
              
              left_index = 0
              right_index = 0 
              nex_sorted = [] 
              for i in range(0, len(left)): 
                
                  if left_index >= len(left):  left is empty , append all the right.
                    new_sorted.append(right[right_index])
                    right_ndex +=  1
                else if right_index >= len(right): 
                    new_sorted.append(right[right_index])
                    right_ndex +=  1

                    
                  if  left[left_index]  <   right[right_index] :
                      new_sorted.append(left[left_index])
                      left_index += 1 
                  else:
                    new_sorted.append(right[right_index])
                    right_ndex +=  1
                            
              return arr

            
    Sequential  iteration and increase index 
    based on condition.












############################################################################################
############################################################################################    
209. Minimum Size Subarray Sum       Medium

            Given an array of positive integers nums and a positive integer target,
                return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr]
                of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.

        Example 1:
            Input: target = 7, nums = [2,3,1,2,4,3]
            Output: 2
            Explanation: The subarray [4,3] has the minimal length under the problem constraint.


        Example 2:
            Input: target = 4, nums = [1,4,4]
            Output: 1


        Example 3:
            Input: target = 11, nums = [1,1,1,1,1,1,1,1]
            Output: 0
            
        Constraints:
            1 <= target <= 109
            1 <= nums.length <= 105
            1 <= nums[i] <= 105
        


        ##### O(N)  
        class Solution:
            def minSubArrayLen(self, s, nums):
                total = left = 0
                result = len(nums) + 1
                
                for right, n in enumerate(nums):
                    total += n   ##### Init with the max one (as we need to take the minimal)
                    while total >= starget :
                        result = min(result, right - left + 1)  #### calc current array
                        total -= nums[left]   ### move on t
                        left += 1    ### Decrement the left to make it smaller.
                        
                return result if result <= len(nums) else 0

            
            iright all array range()
            ileft : decrement until the condition is ok, (we take minimal size)    
            
            just following the right
            
            left is GLOBAL (not local to the loop):
                left=0   --> right
            left --->  [] right
            
                Why left is global :
                
                    left = 
                    while totla > starget:
                    
                array have only +ve integers OK 
                        reassining left to 0 will make you go through the loop from the beginning 
                            We don't need that because we will always skip all numbers before previous left 
                                We are trying ot reach minimal length of the subarray 
                
                why this :  "we will always skip all numbers before previous left "
                    because :
                        while total >= starget :
                        
                        If we include left= [0 ... LeftCurrent] :
                            By definition of  while total >= starget  And array[i]>0 :
                                total[left_i] > starget is ALWAYS
                                condiiton was ALWAYS preserved.
                                
                            we do not need to iterate [0 ..LeftCurrent]     
                                
                            As soon condition is ok, we can keep global left.
                            
                    
                    Iterate on the Conditions.
                    
                    Continuous array is often O(N)  or O(N. Log N)
                    
                    
                    If number Negative >0             
                        will to iterate Left = 0... N  
                    
                    Sorted array 
                                
                    
                    
                Why we dont need to 
                total -= nums[left]   ### move on t

                    
            Condition of continusous array :
                [i0:i1]
                
            


            
        #######  O(n log n)    
        class Solution:
        def minSubArrayLen(self, target, nums):
            result = len(nums) + 1
            for idx, n in enumerate(nums[1:], 1):
                nums[idx] = nums[idx - 1] + n
            left = 0
            for right, n in enumerate(nums):
                if n >= target:
                    left = self.find_left(left, right, nums, target, n)
                    result = min(result, right - left + 1)
            return result if result <= len(nums) else 0

        def find_left(self, left, right, nums, target, n):
            while left < right:
                mid = (left + right) // 2
                if n - nums[mid] >= target:
                    left = mid + 1
                else:
                    right = mid
            return left



    
    
    
    
    
    















#########################################################################################  
#########################################################################################
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
            
            









    
    
##############################################################################################
##############################################################################################
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



































        
    
    