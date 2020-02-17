
/**
 * Write a description of class Recursion1 here.
 * 
 * @Sara Rodriguez y Stiven Yepes
 * @01
 */
public class Recursion1
{
    /** Ejercicio 1
     * Given base and n that are both 1 or more, compute recursively (no loops) the value of base to the n power, so powerN(3, 2) is 9 (3 squared).
     * powerN(3, 1) → 3
     * powerN(3, 2) → 9
     * powerN(3, 3) → 27
     */
    
    public int powerN(int base, int n) 
    {
        if (n==1)
            return base;
        else
            return powerN(base, n-1)*base;
    }
    
    /** Ejercicio 2
     * Given a string, compute recursively (no loops) the number of lowercase 'x' chars in the string.
     * countX("xxhixx") → 4
     * countX("xhixhix") → 3
     * countX("hi") → 0
     */
    
    public int countX(String str) 
    {
        if(str.length() == 0)
		return 0;
		
	if(str.charAt(0) == 'x')
		return 1 + countX(str.substring(1));
		
	return countX(str.substring(1));
    }
    
    /** Ejercicio 3
     * Given a string, compute recursively (no loops) a new string where all appearances of "pi" have been replaced by "3.14".
     * changePi("xpix") → "x3.14x"
     * changePi("pipi") → "3.143.14"
     * changePi("pip") → "3.14p"
     */
    
    public String changePi(String str) 
    {
	if(str.length() < 2)
		return str;
	
	if(str.substring(0, 2).equals("pi"))
		return "3.14" + changePi(str.substring(2));
		
	return str.charAt(0) + changePi(str.substring(1));
    }
    
    /** Ejercicio 4
     * 
     * Given a string, compute recursively a new string where all the adjacent chars are now separated by a "*".
     * allStar("hello") → "h*e*l*l*o"
     * allStar("abc") → "a*b*c"
     * allStar("ab") → "a*b"
     */
    public String allStar(String str) 
    {
        if (str.length()<2)
            return str;
        return str.substring(0,1) + "*" + allStar(str.substring(1));
    }
    
    /** Ejercicio 5
     * Given a string, compute recursively a new string where all the lowercase 'x' chars have been moved to the end of the string.
     * endX("xxre") → "rexx"
     * endX("xxhixx") → "hixxxx"
     * endX("xhixhix") → "hihixxx"
     */
    public String endX(String str) 
    {
        if (str.length() < 1)
            return str;
        if(str.charAt(0) == 'x')
            return endX(str.substring(1)) + "x";
        return str.charAt(0) + endX(str.substring(1));
    }
    

}
