

import java.util.Random;
/**
*
* @author 
*/

public class TestTaller4 
{
      public static boolean ejercicio1(){
	        int a,b,c,d;
	        a = Taller4.arrayMax(new int[] {12,324,43,2,3,43,2,3,43},8);
	        b = Taller4.arrayMax(new int[] {3,2,343,2,43,55,67,68,86,3,4},10);
	        c = Taller4.arrayMax(new int[] {56,7,6,45,8,4,34,8,7,5,34,7,78,9},13);
	        d = Taller4.arrayMax(new int[] {1,2,3,4,5,5},5);
	        
	        if(a!=324 || b!=343 || c!=78 || d!=5)
	            return false;
	        return true;
	    }
	    
	    public static boolean ejercicio2(){
	    	boolean a, b, c, d;
			a=Taller4.groupSum(0, new int[] {2, 4, 8}, 9);
			b=Taller4.groupSum(0, new int[] {2, 4, 8}, 8);
			c=Taller4.groupSum(0, new int[] {10, 2, 2, 5}, 9);
			d=Taller4.groupSum(0, new int[] {10, 2, 2, 5}, 17);
			if(!a && b && c && d)
				return true;
			return false;
		}
	    
	    
	    public static boolean ejercicio3(){
	       long a,b,c,d;
	       a = Taller4.fibonacci(4);
	       b = Taller4.fibonacci(8);
	       c = Taller4.fibonacci(12);
	       d = Taller4.fibonacci(16);
	       if(a==3 && b==21 && c==144 && d==987)
	    	   return true;
	       return false;
	    }
	    
	    public static void main(String[] args)
      {
	        //Ejercicio1
	        if(ejercicio1())
	            System.out.println("Ejercicio 1 Correcto");
	        else
	        System.out.println("Ejercicio 1 Incorrecto");

                testEj1();
	       
	      //Ejercicio2
	        if(ejercicio2())
	            System.out.println("Ejercicio 2 Correcto");
	        else
	        System.out.println("Ejercicio 2 Incorrecto");

                testEj2();
	        
	        //Ejercicio3
	        if(ejercicio3())
	            System.out.println("Ejercicio 3 Correcto");
	        else
	        System.out.println("Ejercicio 3 Incorrecto");

                testEj3(); 
         
	    }

      public static void testEj1()
      {
          
        for(int i=80; i<=100; i++)
          {
            
            int[] arr = new int[i];
            Random rnd = new Random();
            for (int j = 0; j < arr.length; j++) 
            {
            arr[j] = rnd.nextInt(1000 - 1 );
            }
            long inicio= System.currentTimeMillis();
            int a = Taller4.arrayMax(arr, i-1);
            long fin= System.currentTimeMillis();
            long tiempo = fin-inicio;
            System.out.println(i+" "+(fin-inicio));
            
          }
      }

      public static void testEj2()
      {
         for(int i=10; i<=30; i++){
            int[] arr = new int[i];
            Random rnd = new Random();
            for (int j = 0; j < arr.length; j++) 
            {
            arr[j] = rnd.nextInt(100);
            }
            int max = 150, min = 80;
            int target = rnd.nextInt((max - min) + 1) + min;
            long inicio= System.currentTimeMillis();
            boolean a = Taller4.groupSum(0,arr,target);
            long fin= System.currentTimeMillis();
            long tiempo = fin-inicio;
            System.out.println(i+" "+(fin-inicio));
            
          }
      }

      public static void testEj3()
      {
        for(int i=20; i<=40; i++)
         {
            //int fib = 30;
            long inicio= System.currentTimeMillis();
            long a = Taller4.fibonacci(i);
            long fin= System.currentTimeMillis();
            long tiempo = fin-inicio;
            System.out.println(i+" "+(fin-inicio));
            //fib++;
          } 
      }
}
