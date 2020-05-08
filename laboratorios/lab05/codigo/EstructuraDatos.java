import java.util.*;
public class EstructuraDatos
{
    public int size;
    public ArrayList<Triplet<Node, Node, Double>> adjGraph = new ArrayList<>();
    
    public EstructuraDatos(HashMap<Long, Node> nodes, ArrayList<Triplet<Long, Long, Double>> edges){
        this.size = size();
        for(Triplet<Long, Long, Double> t : edges) {
            // Triplet<Node, Node, Double> tr = new Triplet(_____, _____, ____);
            Triplet<Node, Node, Double> tr = new Triplet(new Node(t.getX()), new Node(t.getY()), t.getZ());
            adjGraph.add(tr);
        }
    }

    /**
     * Metodo para obtener una lista de hijos desde un nodo, es decir todos los nodos
     * asociados al nodo pasado como argumento
     * @param vertex nodo al cual se le busca los asociados o hijos
     * @return todos los asociados o hijos del nodo vertex, listados en una ArrayList
     * Para más información de las clases:
     * @see <a href="https://docs.oracle.com/javase/8/docs/api/java/util/ArrayList.html"> Ver documentacion ArrayList </a>
     */
    public  ArrayList<Long> getSuccessors(Long vertexID){
        ArrayList<Long> sucesores = new ArrayList<>();
        //  for(type variableName : arrayName){
        //  for (Integer num : arrlist)
        for( int i=0; i<size;i++ )
        {
            Triplet<Node, Node, Double> n = adjGraph.get(i);
            if(vertexID == n.getX().getID()){
                sucesores.add(n.getY().getID());
            }
        }
        return sucesores;
    }

    /**
     * Metodo para obtener el peso o longitud entre dos nodos
     * 
     * @param source desde donde inicia el arco
     * @param destination  donde termina el arco
     * @return un entero con dicho peso
     */ 
    public Double getWeight(Long sourceID, Long destinationID){
        for(int i=0; i < adjGraph.size(); i++){
            if((sourceID == adjGraph.get(i).getX().getID()) && (destinationID == adjGraph.get(i).getY().getID())){
                return adjGraph.get(i).getZ();
            }
        }
        return -1.0;
    }

    /**
     * Metodo que tiene la intencion de retornar el tamaño del grafo
     * @return tamaño del grafo
     */
    public int size() {
        return this.size;
    }
}