import java.util.ArrayList;
import java.util.LinkedList;

/**
 * Implementacion de un grafo dirigido usando listas de adyacencia
 *
 * @author Mauricio Toro, Mateo Agudelo, <su nombre>
 */
public class DigraphAL extends Digraph {
  
	private ArrayList<LinkedList<Pair<Integer,Integer>>> nodo;

  //Experimental
  //private ArrayList<LinkedList<Pair2>> nodo2;

	/**
	* Constructor para el grafo dirigido
	* @param vertices el numero de vertices que tendra el grafo dirigido
	*
	*/
	public DigraphAL(int size) 
  {
		super(size);
		ArrayList<LinkedList<Pair<Integer,Integer>>> nodo = new ArrayList<>();
        for (int i = 0; i < size + 1; i++) 
        {
            nodo.add(new LinkedList<Pair<Integer,Integer>>()); 
	      }
  }

	/**
	* Metodo para añadir un arco nuevo, donde se representa cada nodo con un entero
	* y se le asigna un peso a la longitud entre un nodo fuente y uno destino	
	* @param source desde donde se hara el arco
	* @param destination hacia donde va el arco
	* @param weight el peso de la longitud entre source y destination
	*/
	public void addArc(int source, int destination, int weight) 
  {
    //Probablemente se inicializó mal en el constructor (arriba)
    //Spoiler: nodo.get(source).add(________ Pair<>(destination,weight));
    nodo.get(source).add(new Pair<>(destination,weight));
		
	}

	/**
	* Metodo para obtener una lista de hijos desde un nodo, es decir todos los nodos
	* asociados al nodo pasado como argumento
	* @param vertex nodo al cual se le busca los asociados o hijos
	* @return todos los asociados o hijos del nodo vertex, listados en una ArrayList
	* Para más información de las clases:
 	* @see <a href="https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html"> Ver documentacion ArrayList </a>
	*/
	public ArrayList<Integer> getSuccessors(int vertex) {
    ArrayList<Integer> n = new ArrayList<>();
        // idek
        nodo.get(vertex).forEach(i -> n.add(i.first));

        //Original
        //nodo.get(vertex).forEach(i -> n.add(i._______));

        return n;
		
	}

	/**
	* Metodo para obtener el peso o longitud entre dos nodos
	* 
	* @param source desde donde inicia el arco
	* @param destination  donde termina el arco
	* @return un entero con dicho peso
	*/	
	public int getWeight(int source, int destination) {
    int result = 0;
        for (Pair<Integer, Integer> integerIntegerPair : nodo.get(source)) {
            if (integerIntegerPair.first == destination ) 
              result = integerIntegerPair.second;
        }
        return result;
		
	}

}