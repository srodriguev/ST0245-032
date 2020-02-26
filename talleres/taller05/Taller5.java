public class Taller5 {  
    
    

    /**
    * @param array es una arreglo de numeros enteros.
    * El método suma tiene la intención de hacer el proceso de suma
    * mediante una funcion cíclica (while/for/...)
    * @return la suma de todos los numeros sumados.
    */
    public static int suma (int[]array){
      int suma=0;
      for(int i=0;i<array.length;i++){
        suma=suma+array[i];
      }
      return suma;
    }
    
    
    /**
    * @param num es el numero el cual se utiliza para ser multiplicado.
    * El método mul tiene la intención de hacer la multiplicación
    * de 1 a n por el numero mul
    * mediante una funcion cíclica (while/for/...)
    * 
    */
    public static void mul (int num){
      for(int i=1; i<=num;i++){
        for(int j=1; j<11; j++){
          System.out.println(i+" x "+j+" = "+(i*j));
        }
      }
    }
    
    
    /**
    * @param array es un arreglo de números desordenados
    * El método insertionSort tiene la intención ordenar los números
    * del arreglo array por el método insertion:
    * @see <a href="https://www.youtube.com/watch?v=OGzPmgsI-pQ"> Insertion Sort <a/>
    * mediante la anidación de funciones cíclicas (while/for/...)
    * 
    */
    public static int[] insertionSort (int[] array){
    int j=0;
	  int temp=0;
	  for(int i=0;i<array.length;i++){
	    j = i;
	    temp = array[j];
	    while(j > 0 && array[j-1] > temp){
		    array[j] = array[j-1];
		    j -- ;
	    }
	    array[j] = temp;
	}
	return array;
    }
}