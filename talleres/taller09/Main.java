import java.util.HashMap;

/**
 * Dar solucion taller numero 9
 * Se recomienda hacer una clase o metodo main que instancie esta clase de manera que sirva como test
 * y desarrolle las actividades y cuestiones presentadas en el taller.
 * En esta clase se hará uso total de la clase HashMap:
 * @see <a href="https://docs.oracle.com/javase/8/docs/api/java/util/HashMap.html">HashMap API</a>
 * @version 1
 * @author Mauricio Toro, Andres Paez
 */
public class Main{
	
  private static HashMap<String, Integer> empresas  = new HashMap<>();
    /** 
    * pedrito 2
    * @param empresas es el hashmap ya creado donde se guardaran los valores 
    * @param key es la llave en el conjunto
    * @param value es el valor asociado a la llave key
    * El metodo agregara un nuevo valor a empresas con los atributos key y value.
    */
    
    public static void agregar(HashMap empresas, String key, String value) {
        empresas.put(key, value);
    }

    /** 
    * pedrito 3
    * @param empresas es el hashmap ya creado donde se guardaran los valores 
    * @param key es la llave en el conjunto
    * 
    * El metodo buscara el valor asociado a la llave key
    * @return true, si lo encuentra, false de lo contrario.
    */
    public static boolean buscar(HashMap empresas, String key) {
        return empresas.containsKey(key);
        
    }
    /** 
    * pedrito 4
    * @param empresas es el hashmap ya creado donde se guardaran los valores 
    * @param value es el valor asociado a la llave key
    * El metodo nos dira si hay una llave asociado al valor value.
    * @return true, si lo encuentra, false de lo contrario.
    */
    public static boolean contieneValue(HashMap empresas, String value) {
        return empresas.containsValue(value);
    }

    public static void main(String[] args){
      agregar(empresas, "001", "Familia");
      agregar(empresas, "002", "Argos");
      agregar(empresas, "003", "Ecopetrol");

      boolean exp = buscar(empresas, "003");
      System.out.println(exp);

      boolean exp2 = contieneValue(empresas,"Familia");
      boolean exp3 = contieneValue(empresas," BMW");
      System.out.println(exp2);
      System.out.println(exp3);
    }
}