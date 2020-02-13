
/**
 * Write a description of class Test here.
 * 
 * @author (your name) 
 * @version (a version number or a date)
 */
public class Test
{
     public static void main(String[] args){
        for (int i = 27; i <= 37; i++){
           long inicio = System.currentTimeMillis();
           f(i);
           long fin = System.currentTimeMillis();
           System.out.println(i + " " + (fin-inicio));
        }
           
    }
    
    public static int f( int n){
        if (n <= 1)
          return n;
        else 
          return f(n-1) + f(n-2);
    }
    
}
