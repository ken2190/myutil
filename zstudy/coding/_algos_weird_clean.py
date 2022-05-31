




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



































        
    
    