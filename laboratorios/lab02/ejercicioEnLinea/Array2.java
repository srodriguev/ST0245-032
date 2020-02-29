
public class Array2 {

    //método que venía por defecto en la plantilla.
    public int countEvensDefault(int[] nums) {
        int count = 0;//        C
        for (int n : nums) {//  n
            if (n % 2 == 0) {// n
                count++;    //  n
            }
        }
        return count;       //C
        //Complejidad O(n)
    }
    
    /**
    * Return the number of even ints in the given array. Note: the % "mod" operator computes  
    *  the remainder, e.g. 5 % 2 is 1.
    */
    //método que implementamos
    public int countEvens(int[] nums) 
      {
         int counter=0;
         for (int i=0; i<nums.length;i++)
         {
             if (nums[i]%2 == 0)
               counter++;
         }
                return counter;
      }

    public int bigDiff(int[] nums) 
    {
      int max = nums[0];
      int min = nums[0];
      for (int i=0; i<nums.length;i++)
      {
        if (nums[i]>max)
            max = nums[i];
          
        if (nums[i]<min)
            min = nums[i];
      }
      return max-min;
    }
    
    public int sum13(int[] nums) 
    {
      int total=0;
      boolean able=true;
      if (nums.length ==0)
        return 0;
      
      for (int i=0;i<nums.length;i++)
      {
        if (nums[i] != 13 && able)
          total += nums[i];
        else
          able=false;
        if (able==false && nums[i]!=13)
          able = true;
      }
      return total;
    }
    
    public boolean has22(int[] nums) 
    {
      for (int i=0; i<nums.length-1;i++)
      {
        if (nums[i] == 2 && nums[i+1]==2)
          return true;
      }
      return false;
    }
    
    public boolean only14(int[] nums) 
    {
      boolean goingGood = true;
      for (int i=0; i<nums.length;i++)
      {
        if (nums[i]!= 1 && nums[i] !=4)
          goingGood = false;
      }
      if (goingGood)
        return true;
      else
        return false;
    }

}