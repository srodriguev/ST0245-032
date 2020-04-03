import java.util.*;

public class NeverasSolicitudes
{
  /**
    * @param neveras es una estructura de datos para representar el almacen con las neveras
    * @param solicitudes es una estructura de datos para representar las solicitudes
    */
    public static void asignarSolicitudes (Stack<String> neveras, Queue<String> solicitudes)
    {
      if (  solicitudes.isEmpty() || neveras.isEmpty() )
      {
        System.out.println("O no hay solicitudes, o no hay neveras");
      }

      while (!solicitudes.isEmpty() && !neveras.isEmpty())
      {
        String actualSolicitud =  solicitudes.poll();
        String[] solElemnts = actualSolicitud.split(",");
        int cantNeveras = Integer.parseInt(solElemnts[1]);
        String[] solNeveras = new String[cantNeveras];

        System.out.println("> Para: "+solElemnts[0]+" hay: ");
        for (int i=0; i<cantNeveras;i++)
        {
          if (!neveras.isEmpty())
          {
            solNeveras[i] = (neveras.pop());
            System.out.print(solNeveras[i]+" ");
          }
          
        }
        System.out.println();
      }
      
    }


    public static void ejercicioNeveras()
    {
      Stack<String> neveras = new Stack<>();
      neveras.push("1,haceb");
      neveras.push("2,lg");
      neveras.push("3,ibm");
      neveras.push("4,haceb");
      neveras.push("5,lg"); 
      neveras.push("6,ibm");
      neveras.push("7,haceb");
      neveras.push("8,lg");
      neveras.push("9,ibm");
      neveras.push("8,lg");
      neveras.push("9,ibm");
      
      Queue<String> pedidos = new LinkedList<>();
      pedidos.add("éxito,1");
      pedidos.add("olimpica,4");
      pedidos.add("la14,2");
      pedidos.add("eafit,10"); 

      System.out.println("Caso 1: ");
      asignarSolicitudes(neveras,pedidos);

      Stack<String> neveras2 = new Stack<>();
      Queue<String> pedidos2 = new LinkedList<>();
      pedidos2.add("carulla,1");

      System.out.println("Caso 2: ");
      asignarSolicitudes(neveras2,pedidos2);

    }


    public static void main(String[] args)
    {
    	
        System.out.println("Neveras y Pedidos: ");
        ejercicioNeveras();
        
    }

}
	