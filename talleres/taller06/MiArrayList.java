import java.util.Arrays;

/**
 * La clase MiArrayList tiene la intención de representar un objeto que simule el comportamiento
 * de la clase ya implementada "ArrayList"
 * es claro que no se puede utilizar dicha estructura ya utilizada dentro de este ejercicio. 
 * Para más información de la clase ArrayList:
 * @see <a href="https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html"> Ver documentacion ArrayList </a>
 * 
 * 
 * @author Mauricio Toro, Andres Paez
 * @version 1
 */

public class MiArrayList {
    private int size;
    private static final int DEFAULT_CAPACITY = 10;
    private int elements[]; 
    
    /**
    * El metodo constructor se utiliza para incializar
    * variables a valores neutros como 0 o null.
    * El contructor no lleva parámetros en este caso.
    */
    public MiArrayList() {
      size = 0;
      elements = new int[DEFAULT_CAPACITY];
    }     

    
    /**
    * Tiene la intención de retornar la longitud del objeto
    * @return longitud del objeto
    *
    * El size esta influenciado por las funciones add y del
    */
    public int size() {
      return size;
    }   
    
        /** 
        * @param e el elemento a guardar
        * Agrega un elemento e a la última posición de la lista
        *
        */   
       public void add(int e)
       {
        try {
          add(size,e);
        }
        catch(Exception Exception) {
          System.out.println("index out of bounds exception");
        }
       }
   
       //v1 code. Se repetía código.
     public void addVersion1(int e) {
       if(size<DEFAULT_CAPACITY)
        elements[size+1]=e;
      else{
        int[] elements2 = new int[size*2];
        for(int i=0; i<size;i++){
          elements2[i]=elements[i];
        }
        elements2[size+1]=e;
        elements = elements2;
        size++;
      }

    }    
    
    
    /** 
    * @param i es un íncide donde se encuentra el elemento posicionado
    * Retorna el elemento que se encuentra en la posición i de la lista.
    *
    */
    public int get(int i) throws Exception 
    {
        if(i<size && i>0)
          return elements[i];
        else 
          throw new Exception("index out of bound exception index = "+i);          
    }
    
    
    /**
    * @param index es la posicion en la cual se va agregar el elemento
    * @param e el elemento a guardar
    * Agrega un elemento e en la posición index de la lista
    * 
    */
    public void add(int index, int e)throws Exception {
       if(index > size+1)
       {
          throw new Exception("index out of bound exception index = "+index);
        }
        else{
          if(size == elements.length)
          {
            int[] elements2 = new int[size*2];
            
            for(int i=0; i<size;i++)
            {
              elements2[i]=elements[i];
            }
              
            elements = elements2;
            
            //corro los elementos necesarios un espacio desde el final hasta el index. Le sumo de una vez el espacio extra.
            size++;
            for(int i=size; i<=index;i--)
            {
               elements[i+1]=elements[i];
            }
            
            elements[index]=e;
            
              }
          else{
            size++;
            for(int i=size; i<=index;i--)
            {
                elements[i+1]=elements[i];
              }
            
            elements[index]=e;
              //size++;
          }
        }
      } 
     

    /**
    * @param index es la posicion en la cual se va eliminar el elemento
    *
    * ELimina el elemento  en la posición index de la lista
    *
    */
    public void del(int index)throws Exception{
      if(index >= size)
      {
        throw new Exception("index out of bound exception index = "+index);
      }
      else
      {
        for(int i=size-1;i>=index;i--)
        {
          elements[i-1]=elements[i];
        }
      }
    }
}