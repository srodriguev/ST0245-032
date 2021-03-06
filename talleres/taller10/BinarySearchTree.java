 public class BinarySearchTree {

     private Node root;
  
    
        // Constructor sin parametros iniciar sin nodo
    public BinarySearchTree() {
        this.root = root;
    }
    //Contructor iniciando con nodo
    public BinarySearchTree(int n) {
        this.root = new Node(n);
    }

    //Llama al metodo auxiliar insertar
    public void insertar(int n) {
        insertarAux(root, n); // O(log2(n))
    }
    // Agrega un nodo al arbol
    // Worst case O(n) - Sin balancear
    //O(logn) - Average - sin balancear
    private void insertarAux(Node node, int n) {
        if (node.data == n){
            return;
        }else if (n > node.data) {
            if (node.right == null) {
                node.right = new Node(n);
            }else {
                insertarAux(node.right, n);
            }
        }else {
            if (node.left == null) {
                node.left = new Node(n);
            }else {
                insertarAux(node.left, n);
            }
        }
    }
  
    // Llama al metodo auxiliar buscar
    public boolean buscar(int n) {
        return buscarAux(root, n); //O(log2(n))
    }
    
    //Busca en el arbol si existe un valor, devuelve true o false, dependiendo de si este o no
    // Worst case: O(n) - Sin balancear
    //O(logn) - Average - sin balancear
    private boolean buscarAux(Node node, int n) {
        if (node.data == n) {
            return true;
        }
        else if (node == null) {
            return false;
        }
        else if (node.right == null && node.left == null){
            return false;
        }
        else if (n > node.data) {
            return buscarAux(node.right, n);
        }
        else return buscarAux(node.left, n);
    }

    //Llama al metodo auxiliar borrar
    public void borrar(int n) {
        borrarAux(root, n); 
    }
    
    //Borra un nodo el arbol
    // O(n) Worst Case - sin balancear
    //O(logn) - Average - sin balancear
     private Node borrarAux(Node node, int n) {
        if (node == null) {
            return null;
        }
        if (node.data == n) {
            if (node.right == null && node.left == null) {
                return null;
            }
            if (node.right == null) {
                return node.left;
            }
            if (node.left == null) {
                return node.right;
            }else {
                node.data = encontrarNodoReemplazo(node.left);
            }
        }
        if (n > node.data) {
            node.right = borrarAux(node.right, n);
            return node;
        }
        node.left = borrarAux(node.left, n);
        return node;
    }

    private int encontrarNodoReemplazo(Node n) {
        if (n.right == null) {
            int res = n.data;
            n = null;
            return res;
        }
        return encontrarNodoReemplazo(n.right);
    }


    //Imprimir el arbol
    public void imprimirarbol(BinarySearchTree arbol)
    {
      Node raiz = arbol.root;
      imprimirarbol(raiz);
    }

    //Aux, imprime desde un nodo
    //O(n), recorre todo el árbol
    public void imprimirarbol(Node nodo)
    {
      Node raiz = nodo;
      Node izq = raiz.left;
      Node der = raiz.right;
      if (izq !=null && der !=null)
        System.out.println("Nodo: "+raiz.data+", Left: "+izq.data+" Right: "+der.data);
      else if (izq !=null)
        System.out.println("Nodo: "+raiz.data+" Left: "+izq.data);
      else if (der !=null)
        System.out.println("Nodo: "+raiz.data+" Right: "+der.data);
      else
        System.out.println("Nodo hoja: "+raiz.data);

      if (izq != null)
      {
        imprimirarbol(izq);
      }
      if (der !=null)
      {
        imprimirarbol(der);
      }
      
    }
}