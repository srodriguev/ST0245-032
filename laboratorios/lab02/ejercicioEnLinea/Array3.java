
public class Array3 {

public int maxSpan(int[] nums) 
{
    if (nums.length > 0) 
    {
        int maxSpan = 1;
        for (int i = 0; i < nums.length; i++)
            for (int j = nums.length - 1; j > i; j--)
                if (nums[j] == nums[i]) {
                    int count = (j - i) + 1;
                    if (count > maxSpan) 
                        maxSpan = count;
                    break;
                }
        return maxSpan;
    } else 
        return 0;
}


    public int[] fix34(int[] nums) {
    
      for(int i=0;i<nums.length;i++)
      {
        //checking for 4
        if(nums[i]==4)
        {
          //looping again through the loop
          for(int j=0;j<nums.length;j++)
          {
             //checking for 3
            if(nums[j]==3)
            {
              //swapping 
              int temp=nums[i];
              nums[i]=nums[j+1];
              nums[j+1]=temp;
              }
    
          }
        }
      }
      return nums;
    }
    
    
    public int[] squareUp(int n) 
    {
      int nums[] = new int[n*n];
      int a = n;
      for(int i = 0; i < n; i++) 
      {
        int pos = n*n - i - 1;
        for(int j = 0; j < a; j++) 
        {
          nums[pos -n*j] = i+1;
        }
        a--;
      }
      return nums;
    }
    
    public boolean canBalance(int[] nums) 
    {
        int left = 0;
        int right;
        
        for(int i = 0; i < nums.length - 1; i++)
            left += nums[i];
        
        right = nums[nums.length-1];
        
        for(int i = nums.length - 2; i > 0; i--)
        {
            if(left == right)
                return true;
            left -= nums[i];
            right += nums[i];
        }
        return (left == right);
    }
    
    public boolean linearIn(int[] outer, int[] inner) 
    {
        int indexInner = 0;
        int indexOuter = 0;
        
        while (indexInner < inner.length && indexOuter < outer.length) {
            
            if (outer[indexOuter] == inner[indexInner]) {
                indexOuter++;
                indexInner++;
            } 
            else indexOuter++;
        }
        return (indexInner == inner.length);
    }
    
    public int[] fix45(int[] nums) 
    {
        int[] otherValues = new int[nums.length];
        
          for(int i = 0, c = 0; i < nums.length; i++)
            if(nums[i] != 4 && nums[i] != 5)
              otherValues[c++] = nums[i];
        
          for(int i = 0, c = 0; i < nums.length; i++)
            if(nums[i] == 4)
              nums[++i] = 5;
            else
              nums[i] = otherValues[c++];
        
          return nums;
    }
}
