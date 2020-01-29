        
 /**
 *La clase taller dos tiene como objetivo dar solución
 *a 3 problemas recursivos
 *
 *@autor Mauricio Toro, Camilo Paez
 *@version 1
 */ 
        
  public class Taller2 
  {
            
     /**
            * @param p entrada 1 entero positivo, mayor que q
            * @param q entrada 2 entero positivo, menor que p
            *
            * El método gcd tiene como objetivo ecnontrar el
            * máximo común divisor de dos números, por medio del
            * algoritmo de euclides
            * @see <a href="https://www.youtube.com/watch?v=Q9HjeFD62Uk"> Explicación </a>
            * @see <a href="https://visualgo.net/en/recursion"> Funcionamiento </a>
            *
            * @return el máximo común divisor
            */
     public static int gcd(int p, int q)
     {
                //...
                int res= p % q;
                
                if (res  ==  0)
                {
                    return q;
                }
                else
                {
                    return gcd(q,res);
                }
     }
        
     /**
            * @param nums entrada 2 arreglo de enteros positivos, sobre el cual vamos a interar 
            * @param target entrada 3 entero positivo, determina el valor de referencia 
            * El método SumaGrupo tiene como objetivo darnos a conocer si hay 
            * algun subconjunto el cual su suma = target.
            * 
            *
            * @return verdadero si hay un subconjunto el cual su suma = target
            */
     
     public static boolean SumaGrupo( int[] nums, int target) 
     {
         return SumaGrupo(0, nums, target);
        }
            
     /**
            * @param start entrada 1 entero positivo, determina un índice dentro del proceso
            * @param nums entrada 2 arreglo de enteros positivos, sobre el cual vamos a interar 
            * @param target entrada 3 entero positivo, determina el valor de referencia 
            * El método SumaGrupo tiene como objetivo darnos a conocer si hay 
            * algun subconjunto el cual su suma = target.
            * 
            * Este método SumaGrupo es "private" de modo que solo se puede llamar desde el interior de la clase pues
            * el método que lo representa es el SumaGrupo público.
            * Para más detalles sobre modificadores de acceso:
            * @see <a href="http://ayudasprogramacionweb.blogspot.com/2013/02/modificadores-acceso-public-protected-private-java.html"> modificadores </a>
            *
            *
            * @return verdadero si hay un subconjunto el cual su suma = target
            * 
            * N O T A S :
            * Tenemos dos versiones que pudimos llegar como grupo. Una mas extensa y otra aún más resumida.
            * También al consultar más sobre el tema hemos encontrado un método con tablas de hallar la suma, se anexa como dato curioso
            * 
            */
     private static boolean SumaGrupo(int start, int[] nums, int target) 
      {
            if(target == 0)
                 return true;
            
            if(start == nums.length)
                 return false;
            
            if(SumaGrupo(start + 1, nums, target - nums[start]))
                 return true;
            
            return SumaGrupo(start + 1, nums, target);
      }
      
     private static boolean SumaGrupo2(int start, int[] nums, int target) 
     {
        
        if(start>=nums.length) 
            return target==0;
        else 
            return (SumaGrupo(start+1, nums, target-nums[start]) || SumaGrupo(start+1, nums, target));
        
    } 
    
    public static boolean SumaGrupoPorTablas(int[] nums, int target) 
      {
        // Created and originally implemented by Tushar Roy (2015) Backtracking and Dynamic Programming Tables. Available watch on: https://www.youtube.com/watch?v=s6FhG--P7z0 as of 28/01/2020.
          
        boolean T[][] = new boolean[nums.length + 1][target + 1];
        
        for (int i = 0; i <= nums.length; i++) 
        {
            T[i][0] = true;
        }

        for (int i = 1; i <= nums.length; i++) 
        {
            for (int j = 1; j <= target; j++) 
            {
                if (j - nums[i - 1] >= 0) 
                {
                    T[i][j] = T[i - 1][j] || T[i - 1][j - nums[i - 1]];
                } else 
                {
                    T[i][j] = T[i-1][j];
                }
            }
        }
        return T[nums.length][target];
      }
            
     /**
            * @param s se trata de una cadena de caracteres sobre la cual hallaremos las posibles combinaciones.
            *
            * El método combinations se define para que solo se tenga que pasar el parametro s y no la cadena 
            * vacía necesaria para el metodo reursivo combinationsAux. Este metodo no se modifica.
            * 
            */
     public static void combinations(String s) 
     { 
                combinationsAux("", s); 
     }
            
     /**
            * @param prefix, se utiliza como una variable auxiliar para guardar datos sobre el proceso.
            * @param s se trata de una cadena de caracteres sobre la cual hallaremos las posibles combinaciones.
            *
            *
            * El método combinationsAux se encarga de encontrar las posibles combinaciones en la cadena s
            * notese que el método es "private" de modo que solo se puede llamar desde el interior de la clase pues
            * el método que lo representa es combinations.
        * Para más detalles sobre modificadores de acceso:
        * @see <a href="http://ayudasprogramacionweb.blogspot.com/2013/02/modificadores-acceso-public-protected-private-java.html"> modificadores </a>
        *
        */
    
       private static void combinationsAux(String prefix, String s) 
       {  
           if(s.length() == 0) 
           {
               System.out.println(prefix);
            }
            else
            {
                combinationsAux(prefix + s.charAt(0), s.substring(1));
                combinationsAux(prefix, s.substring( 1 ));
            }
       }

      /** Not what is requested but put here for experimenting purposes, permutation of a string.
        * Brings far way more results than the suggested combination, Wanted to check how different results would be.
        */
     private static void combinationsAux2(String prefix, String s) 
     { 
         
         if (s.length() == 0)
         { 
             System.out.println(prefix);
          }
         else 
         {
             for (int i = 0; i < s.length(); i++)
             {
                 combinationsAux2(prefix + s.charAt(i), s.substring(0, i) + s.substring(i+1, s.length()));
             }
           }

        }
}