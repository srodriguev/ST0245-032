import java.util.ArrayList;
public class DigraphAM extends Graph
{
    private int size;
    //Grafo con matriz
    int[][] mat;
    public DigraphAM(int size){
        super(size);
        mat = new int[size][size];
        
        for(int i=1; i<=size; i++){
            mat[i][0] = i;
            mat[0][i] = i;
        }
    }

    public int getSize()
    {
      return size;
    }
    
     public void addArc(int source, int destination, int weight){
      mat[source][destination] = weight;
   }
    
    
    public int getWeight(int source, int destination){
       return mat[source][destination];
    }


     public ArrayList<Integer> getSuccessors(int vertex)
   {
       ArrayList<Integer> np= new ArrayList<>();
     for(int i=1; i<=size; i++){
        if(mat[vertex][i]!=0){
         np.add(mat[0][i]);
        }
      }
     return np;   
   }

  public boolean DFS(DigraphAM graph, int i, int j)
  {
    int s = graph.getSize();
    boolean[] visitados = new boolean[s*s];
    return DFSAux(graph,i,j,visitados);
  }

  private boolean DFSAux(DigraphAM graph, int i, int j, boolean[] visitados)
  {
    visitados[i]= true;
    if (i==j)
      return true;
    else{
      for (Integer sus : graph.getSuccessors(i))
      {
        if (!visitados[sus] )
        {
          if (DFSAux(graph,sus,j,visitados))
            return true;
        }
      }
    }

    return false;
  }

  public boolean BFS(DigraphAM graph, int i, int j)
  {
    int s = graph.getSize();
    boolean[] visitados = new boolean[s*s];
    return BFSAux(graph,i,j,visitados);
  }

  private boolean BFSAux(DigraphAM graph, int i, int j, boolean[] visitados)
  {
    visitados[i]= true;
    
    for (Integer sus : graph.getSuccessors(i))
    {
      if (sus==j)
        return true;
    }
    for (Integer sus : graph.getSuccessors(i))
    {
      BFSAux(graph,sus,j,visitados);
    }    

    return false;
  }


}