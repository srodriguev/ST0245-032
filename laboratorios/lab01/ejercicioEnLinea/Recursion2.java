
/**
 * Write a description of class Recursion2 here.
 * 
 * @Sara Rodriguez y Stiven Yepes
 * @01
 */
public class Recursion2
{
    //  A menos que se especifiqué lo contrario estos problemas fueron resueltos usando las guías/tips vistos en Fundamentos de programación, 
    // especificamente en preparación del exámen final de la asignatura con el docente Mauricio en el semestre 2019-2.
    
    /**
     * Ejercicio 1
     * Given an array of ints, is it possible to choose a group of some of the ints, such that the group sums to the given target with this additional constraint: 
     * If a value in the array is chosen to be in the group, 
     * the value immediately following it in the array must not be chosen. (No loops needed.)
     * groupNoAdj(0, [2, 5, 10, 4], 12) → true
     * groupNoAdj(0, [2, 5, 10, 4], 14) → false
     * groupNoAdj(0, [2, 5, 10, 4], 7) → false
     */
    public boolean groupNoAdj(int start, int[] nums, int target) 
    {
        if(target == 0)
             return true;
  	if(start >= nums.length)
  	     return false;
  	if(groupNoAdj(start + 2, nums, target - nums[start]))
  	     return true;
  	     
  	return groupNoAdj(start + 1, nums, target);
    }
    
    /** Ejercicio 2
     * Given an array of ints, is it possible to divide the ints into two groups, so that the sums of the two groups are the same. Every int must be in one group or the other. 
     * Write a recursive helper method that takes whatever arguments you like, and make the initial call to your recursive helper from splitArray(). (No loops needed.)
     * splitArray([2, 2]) → true
     * splitArray([2, 3]) → false
     * splitArray([5, 2, 3]) → true
     */
    public boolean splitArray(int[] nums)
    {
        return splitAuxiliar(nums, 0, 0);	
    }

     // recursive helper methodin
    public boolean splitAuxiliar(int[] numeros, int i, int balance)
    {
	if(i == numeros.length)
		return (balance == 0);
	
	if(splitAuxiliar(numeros, i + 1, balance + numeros[i]))
		return true;
		
	return splitAuxiliar(numeros, i + 1, balance - numeros[i]);
    }
    
    /** Ejercicio 3
     * Given an array of ints, is it possible to choose a group of some of the ints, beginning at the start index, such that the group sums to the given target? 
     * However, with the additional constraint that all 6's must be chosen. (No loops needed.)
     */
    public boolean groupSum6(int start, int[] nums, int target) 
    {
        if(start == nums.length)
 	{
 	    if(target == 0)
 	    {
  		 return true;
  	    } else{
  		 return false;
  		  }
  	    
 	}
 	
 	if(nums[start] == 6)
 	  return groupSum6(start + 1, nums, target - nums[start]);
  
        if(groupSum6(start + 1, nums, target - nums[start]))
            return true;
  
        return groupSum6(start + 1, nums, target);
   }
  
  /** Ejercicio 4
   * Given an array of ints, is it possible to choose a group of some of the ints, such that the group sums to the given target, with this additional constraint: 
   * if there are numbers in the array that are adjacent and the identical value, they must either all be chosen, or none of them chosen. For example, with the array {1, 2, 2, 2, 5, 2}, 
   * either all three 2's in the middle must be chosen or not, all as a group. (one loop can be used to find the extent of the identical values).


   * groupSumClump(0, [2, 4, 8], 10) → true
   * groupSumClump(0, [1, 2, 4, 8, 1], 14) → true
   * groupSumClump(0, [2, 4, 4, 8], 14) → false
   * 
   * Algoritmo de respuesta analizado con ayuda del algoritmo propuesto por Mirandaio:
   * 
   * Title: Group Sum Clump
   * Author: Mirandaio
   * Date: 2014
   * Code version: 2.0
   * Availability: https://github.com/mirandaio/codingbat/blob/master/java/recursion-2/groupSumClump.java
   * 
   */
  public boolean groupSumClump(int start, int[] nums, int target) 
 {
    if(start >= nums.length)
        return target == 0;
          
    int i = start;
    int sum = 0;
    
    while(i < nums.length && nums[start] == nums[i]) {
        sum += nums[i];
        i++;
    }
                              
    if(groupSumClump(i, nums, target - sum))
        return true;
                                        
    if(groupSumClump(i, nums, target))
        return true;
                                                  
    return false;
 }

/** Ejercicio 5
 * Given an array of ints, is it possible to divide the ints into two groups, so that the sum of one group is a multiple of 10, and the sum of the other group is odd. Every int must be in one group or the other. 
 * Write a recursive helper method that takes whatever arguments you like, and make the initial call to your recursive helper from splitOdd10(). (No loops needed.)
 * 
 * Código realizado en clase de Fundamentos de Programación.

 * splitOdd10([5, 5, 5]) → true
 * splitOdd10([5, 5, 6]) → false
 * splitOdd10([5, 5, 6, 1]) → true
 */
public boolean splitOdd10(int[] nums) 
{
  return isSideOdd(nums, 0, 0, 0);	
}

// recursive helper method
public boolean isSideOdd(int[] nums, int i, int group1, int group2)
{
	if(i == nums.length)
		return (group1 % 2 == 1 && group2 % 10 == 0 || group2 % 2 == 1 && group1 % 10 == 0);
	
	if(isSideOdd(nums, i + 1, group1 + nums[i], group2))
		return true;
	return isSideOdd(nums, i + 1, group1, group2 + nums[i]);
}

}
