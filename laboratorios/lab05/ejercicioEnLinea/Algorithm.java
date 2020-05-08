 import java.util.*;
 
 public class Algorithm 
{
  public static boolean[] visited;
  public static int[] colors;
  public static ArrayList<Integer> ady = new ArrayList<Integer>();
  public static DigraphAM graph;
  public static LinkedList<Integer> cola= new LinkedList<Integer>();

    // public static boolean DFSColorFC(Tree tree) < así venía la plantilla
    public static boolean DFSColorFC(DigraphAM g) {
        graph = g;
        visited = new boolean[graph.size()];
        colors = new int[graph.size()];
        for(int i=0; i<graph.size();i++) colors[i]=-1;
        
        return DFSColorFCAux(graph.getFirst());
    }

    private static boolean DFSColorFCAux(int n) {
     visited[n]=true;
     colors[n]=1;
     cola.add(n);
     ady = graph.getSuccessors(n);

    while(cola.size() != 0){
      int u = cola.poll();
  
     for(int i=0; i<ady.size()-1;i++){
       
       int sig = ady.get(i);

        if (colors[n]==colors[sig] && visited[sig]){
           return false;
        }
       else if(!visited[sig]){
         DFSColorFCAux(sig);
       }
     }
    }
       return true;
    }
}